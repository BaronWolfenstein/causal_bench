"""Fieller & MOVER ratio confidence intervals (ratio_ci) — win-ratio CI with the correct U-statistic variance.

Validates: (a) the DeLong U-statistic SE matches a bootstrap; (b) Fieller ≈ delta at a large denominator but is
asymmetric/wider at a small one (the heavy-tail robustness Fieller is for); (c) the degenerate 'exterior'/
'unbounded' Fieller cases are returned rather than crashing; (d) MOVER sanity."""
import numpy as np

from causal_bench.validation.ratio_ci import win_ratio_ci, fieller_ci, delta_ratio_ci, mover_difference


def _wr_samples(shift, m=300, seed=0):
    g = np.random.default_rng(seed)
    return g.normal(shift, 1, m), g.normal(0, 1, m)


def test_ustat_se_matches_bootstrap():
    t, c = _wr_samples(0.5, 300, seed=0)
    r = win_ratio_ci(t, c, method="delta")
    se_delta = (r["hi"] - r["lo"]) / (2 * 1.96)
    g = np.random.default_rng(1)
    bs = []
    for _ in range(1500):
        ti = g.integers(0, 300, 300); ci = g.integers(0, 300, 300)
        tt = t[ti][:, None]; cc = c[ci][None, :]
        bs.append((tt > cc).mean() / ((tt < cc).mean() + 1e-12))
    assert abs(se_delta - np.std(bs)) / np.std(bs) < 0.15      # within 15% of the bootstrap SE


def test_fieller_agrees_at_large_denominator():
    t, c = _wr_samples(0.3, 250, seed=2)                        # many losses ⇒ denominator well-identified
    f = win_ratio_ci(t, c, method="fieller"); d = win_ratio_ci(t, c, method="delta")
    assert f["status"] == "bounded"
    assert abs(f["lo"] - d["lo"]) < 0.1 and abs(f["hi"] - d["hi"]) < 0.1


def test_fieller_wider_upper_at_small_denominator():
    t, c = _wr_samples(2.5, 250, seed=3)                        # few losses ⇒ small denominator
    f = win_ratio_ci(t, c, method="fieller"); d = win_ratio_ci(t, c, method="delta")
    assert f["p_loss"] < 0.1
    assert f["hi"] > d["hi"]                                    # Fieller widens the upper tail
    assert (f["hi"] - f["win_ratio"]) > (f["win_ratio"] - f["lo"])  # and is asymmetric (longer upper arm)


def test_fieller_degenerate_cases_do_not_crash():
    # denominator not significantly != 0 (den small vs its SD): Fieller set is exterior/unbounded, not an error
    pt, lo, hi, status = fieller_ci(num=1.0, den=0.1, v_num=0.01, v_den=1.0, cov=0.0)
    assert status in ("exterior", "unbounded")
    assert np.isfinite(pt)


def test_mover_difference_sanity():
    # difference of two ratios R1=2 (CI [1.5,2.6]) and R2=1 (CI [0.7,1.4]); diff ~1, CI must contain it
    d, L, U = mover_difference(2.0, 1.5, 2.6, 1.0, 0.7, 1.4)
    assert abs(d - 1.0) < 1e-9 and L < 1.0 < U


def test_singleton_clusters_reduce_to_delong():
    # passing each observation as its own cluster must reproduce the independent DeLong CI EXACTLY
    t, c = _wr_samples(0.5, 200, seed=7)
    base = win_ratio_ci(t, c, method="fieller")
    clus = win_ratio_ci(t, c, method="fieller",
                         cluster_treat=np.arange(len(t)), cluster_control=np.arange(len(c)))
    assert abs(clus["var_win"] - base["var_win"]) < 1e-12
    assert abs(clus["var_loss"] - base["var_loss"]) < 1e-12
    assert abs(clus["cov"] - base["cov"]) < 1e-12
    assert abs(clus["lo"] - base["lo"]) < 1e-9 and abs(clus["hi"] - base["hi"]) < 1e-9


def test_cluster_robust_inflates_variance_under_within_cluster_correlation():
    # repeated measures: 100 treated & 100 control SUBJECTS, each observed r=4 times (perfectly correlated).
    # Ignoring clusters pretends 400 independent obs (SE too small); the subject-clustered variance recovers
    # the ~100-subject information and must be substantially larger — same point estimate either way.
    t0, c0 = _wr_samples(0.5, 100, seed=11)
    r = 4
    t = np.repeat(t0, r); c = np.repeat(c0, r)
    ct = np.repeat(np.arange(100), r); cc = np.repeat(np.arange(100), r)
    naive = win_ratio_ci(t, c, method="delta")
    clus = win_ratio_ci(t, c, method="delta", cluster_treat=ct, cluster_control=cc)
    assert abs(naive["win_ratio"] - clus["win_ratio"]) < 1e-12          # duplication doesn't move the point
    assert clus["var_win"] > 2.0 * naive["var_win"]                     # clustered variance is much larger
    assert clus["n_clusters_treat"] == 100 and clus["n_clusters_control"] == 100
    assert (clus["hi"] - clus["lo"]) > 1.5 * (naive["hi"] - naive["lo"])  # and the CI is correspondingly wider


def test_coverage_near_nominal():
    g = np.random.default_rng(99); a = g.normal(0.6, 1, 400000); b = g.normal(0, 1, 400000)
    wr_true = (a > b).mean() / (a < b).mean()
    cov = 0; N = 300
    for s in range(N):
        t, c = _wr_samples(0.6, 200, seed=1000 + s)
        f = win_ratio_ci(t, c, method="fieller")
        cov += f["lo"] <= wr_true <= f["hi"]
    assert 0.90 <= cov / N <= 0.98                              # ~95% nominal


if __name__ == "__main__":
    import pytest, sys
    sys.exit(pytest.main([__file__, "-v"]))
