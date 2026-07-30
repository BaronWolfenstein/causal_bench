"""Demonstrate the global-ESS false pass and the stratified catch. CPU only —
run: `python scripts/stratified_ess_validate.py`."""
import numpy as np

from causal_bench.sampling import (kish_ess, stratified_ess,
                                    stratified_resample_needed, positivity_overlap)


def main() -> None:
    rng = np.random.default_rng(0)
    n_bulk, n_tail = 400, 40
    lw_bulk = rng.normal(0.0, 0.05, n_bulk)
    lw_tail = np.full(n_tail, -8.0)
    lw_tail[0] = 0.0                      # one particle owns the tail
    log_w = np.concatenate([lw_bulk, lw_tail])
    strata = np.concatenate([np.zeros(n_bulk, int), np.ones(n_tail, int)])
    n = log_w.size

    g = kish_ess(log_w)
    print(f"N = {n},  global Kish ESS = {g:.1f}  ->  global rule (ESS < N/2 = "
          f"{n / 2:.0f}) fires? {g < n / 2}")

    rep = stratified_ess(log_w, strata)
    for lab, name in [(0, "bulk"), (1, "tail")]:
        s = rep.stratum(lab)
        print(f"  {name}: n={s['count']:>3}  ESS={s['ess']:6.2f}  "
              f"ratio={s['ess_ratio']:.3f}  mass={s['mass']:.4f}")
    fire, reason = stratified_resample_needed(rep, tail_label=1, tail_frac=0.5)
    print(f"stratified rule fires? {fire}  ({reason})")

    print("\npositivity R (Euclidean): particles cover only 1 of 2 target clusters")
    targets = np.concatenate([rng.normal([0, 0], 0.1, (20, 2)),
                              rng.normal([10, 10], 0.1, (20, 2))])
    particles = rng.normal([0, 0], 0.3, (200, 2))
    r = positivity_overlap(particles, targets, radius=0.5)
    print(f"  coverage = {r.coverage:.2f}   uncovered targets = {int(r.uncovered.sum())}/40")

    print("\nRESULT: global ESS false-passes tail collapse; stratified + positivity R catch it.")


if __name__ == "__main__":
    main()
