"""python experiments/demo_smc_ipcw.py — SMC + IPCW on a far target, CPU.

End-to-end: run twisted-SMC to a rare (far-from-mass) target, report the ESS
trajectory and resample-trigger rate, then apply an informative validity filter
and show that IPCW reweighting + positivity flagging handle the out-of-band kills.
All numpy, no GPU — the algorithmic risks this exercises are hardware-independent.
"""
import numpy as np

from causal_bench.sampling import run_smc, kish_ess
from causal_bench.sampling.diagnostics import resample_trigger_rate, lineage_multiplicity
from causal_bench.sampling.ipcw import ipcw_weights, positivity_floor
from causal_bench.diagnostics.localization import lineage_collapse_score


def main():
    rng = np.random.default_rng(0)
    mu = np.array([4.0, 0.0])            # rare region: 4 sigma from base mass
    betas = np.linspace(0, 1, 20)        # annealing schedule
    x0 = rng.standard_normal((300, 2))
    prop = lambda x, s: x + 0.3 * np.random.default_rng(s).standard_normal(x.shape)
    lw = lambda x, s: (betas[s] - betas[s - 1]) * (
        -0.5 * ((x - mu) ** 2).sum(1) + 0.5 * (x ** 2).sum(1)
    )

    res = run_smc(x0, prop, lw, len(betas), rng)
    print("trigger rate:", round(resample_trigger_rate(res), 3))
    print("final ESS:   ", round(kish_ess(res.state.log_weights), 1))
    if res.n_resamples:
        mult = lineage_multiplicity(res)
        print("lineage multiplicity (top 5):", sorted(mult)[-5:])
        print("lineage-collapse score:", round(lineage_collapse_score(mult), 3))

    # informative validity filter: keep-prob depends on the covariate -> IPCW
    G = 1.0 / (1.0 + np.exp(-(res.state.particles[:, 0])))
    Gc, viol = positivity_floor(G, floor=1e-3)
    print("positivity violations:", int(viol.sum()))
    print("mean IPCW weight:     ", round(float(ipcw_weights(Gc).mean()), 3))


if __name__ == "__main__":
    main()
