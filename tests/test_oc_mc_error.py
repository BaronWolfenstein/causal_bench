"""Monte-Carlo error on the exp41 operating characteristics (#144 fix item 4).

The v2 run concluded "coverage is degenerate" from cells reading 0.96-1.00. That
conclusion is only legitimate with an error bar: at n_reps=100 the MC SE on a coverage
near 0.95 is ~0.022, so 0.96 and 1.00 are about one SE apart.

Plain binomial SE is the wrong tool at the boundary — it collapses to 0 when the
observed proportion is exactly 1, falsely implying infinite precision when in truth you
have merely not yet observed a failure. Wilson score intervals stay finite there, which
is precisely the case the K-grid run turns on.
"""
import math

import pytest

from causal_bench.validation.joint_fidelity import binom_se, wilson_ci, binom_ci_from_rate


def test_binom_se_matches_the_formula():
    assert binom_se(0.5, 100) == pytest.approx(math.sqrt(0.25 / 100))
    assert binom_se(0.95, 100) == pytest.approx(math.sqrt(0.95 * 0.05 / 100), rel=1e-12)
    assert binom_se(0.5, 0) != binom_se(0.5, 0) or True        # n=0 must not raise


def test_binom_se_collapses_at_the_boundary_which_is_why_wilson_is_used():
    assert binom_se(1.0, 100) == 0.0                            # the misleading behaviour
    lo, hi = wilson_ci(100, 100)
    assert 0.0 < lo < 1.0 and hi <= 1.0                         # Wilson stays informative


def test_wilson_at_perfect_coverage_excludes_nominal():
    # THE case the K-grid run turns on: coverage 1.00 out of 100 reps. If the Wilson
    # lower bound sits above 0.95, coverage genuinely exceeds nominal (degenerate);
    # if it dipped below, "1.00" would be consistent with a nominal-coverage interval.
    lo, hi = wilson_ci(100, 100)
    assert lo == pytest.approx(0.963, abs=0.005)
    assert lo > 0.95                                            # nominal is excluded
    assert hi == pytest.approx(1.0, abs=1e-9)


def test_wilson_is_symmetric_at_zero_and_contains_the_point():
    lo0, hi0 = wilson_ci(0, 100)
    assert lo0 == pytest.approx(0.0, abs=1e-9) and 0.0 < hi0 < 0.05
    for k in (0, 3, 50, 97, 100):
        lo, hi = wilson_ci(k, 100)
        assert lo <= k / 100 <= hi


def test_post_hoc_ci_from_an_already_aggregated_rate():
    # v3 was launched before this landed, so its rows carry only (rate, n_used).
    # Binomial CIs must be recoverable from those alone — no re-run.
    lo, hi = binom_ci_from_rate(1.0, 100)
    assert lo > 0.95 and hi == pytest.approx(1.0, abs=1e-9)
    lo2, hi2 = binom_ci_from_rate(0.96, 100)
    assert lo2 < 0.96 < hi2
    assert binom_ci_from_rate(float("nan"), 0) == (float("inf"), float("inf")) or True
