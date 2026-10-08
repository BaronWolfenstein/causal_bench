"""Exp 58: CalibrationMonitor — an anytime-valid drift detector for a production calibration/accuracy stream.

The rigorous core of the drift stack (docs/plans/2026-09-24-drift-validity-monitoring-stack.md): a mixture
betting test-martingale (e-process) over the labeled audit stream. You may peek after every observation and refit
on the first alarm, with a UNIFORM false-alarm guarantee (Ville) under a valid fixed baseline and
conditional-mean null. This experiment estimates finite-horizon false alarms on noisy in-control
streams and records post-change FIRST alarms separately from early alarms and misses. Detection
delay and classical thresholds are workload-dependent; this is not a parameter-selection study.
Run: python -m experiments.exp58_calibration_monitor
"""
from pathlib import Path
import numpy as np

from causal_bench.validation.calibration_monitor import (
    run_monitor, naive_repeated_test, simulate_loss_stream, page_hinkley, ddm, simulate_error_stream,
)

OUT_DIR = Path("results/exp58_calibration_monitor")


def alarm_outcome(index,change_at):
    """Alarm indices are 1-based; change_at is the first changed 0-based input."""
    if index is None:return 'missed',None
    if index<=change_at:return 'early',None
    return 'detected',index-change_at


def run(*, mu0=0.15, mu1=0.30, alpha=0.05, n=2000, change_at=500, n_fa=400, n_det=200,
        ph_delta=0.05, ph_lam=6.0, ph_min_n=1):
    if not 0<change_at<n or n_fa<1 or n_det<1:raise ValueError('Positive panel sizes and interior change point required')
    fa_mon = fa_naive = fa_ph = 0
    for seed in range(n_fa):
        x=simulate_loss_stream(n,mu0,seed=seed)
        fa_mon+=run_monitor(x,mu0,alpha=alpha)[1] is not None
        fa_naive+=naive_repeated_test(x,mu0,alpha=alpha) is not None
        fa_ph+=page_hinkley(x,delta=ph_delta,lam=ph_lam,min_n=ph_min_n) is not None
    outcomes={name:dict(detected=0,early=0,missed=0,delays=[]) for name in ('monitor','ph','ddm')}
    def record(name,index):
        kind,delay=alarm_outcome(index,change_at);outcomes[name][kind]+=1
        if delay is not None:outcomes[name]['delays'].append(delay)
    for seed in range(n_det):
        x=simulate_loss_stream(n,mu0,mu1=mu1,change_at=change_at,seed=10_000+seed)
        record('monitor',run_monitor(x,mu0,alpha=alpha)[1])
        record('ph',page_hinkley(x,delta=ph_delta,lam=ph_lam,min_n=ph_min_n))
        record('ddm',ddm(simulate_error_stream(n,mu0,p1=mu1,change_at=change_at,seed=30_000+seed))[1])
    ddm_fa=np.mean([ddm(simulate_error_stream(n,mu0,seed=seed))[1] is not None for seed in range(n_fa)])
    def median(name):
        values=outcomes[name]['delays'];return float(np.median(values)) if values else float('nan')
    return dict(mu0=mu0,mu1=mu1,alpha=alpha,n=n,change_at=change_at,n_fa=n_fa,n_det=n_det,
        ph_delta=ph_delta,ph_lam=ph_lam,ph_min_n=ph_min_n,fa_monitor=fa_mon/n_fa,fa_naive=fa_naive/n_fa,
        fa_ph=fa_ph/n_fa,detected=outcomes['monitor']['detected'],missed=outcomes['monitor']['missed'],
        early_alarms=outcomes['monitor']['early'],median_delay=median('monitor'),
        ph_detected=outcomes['ph']['detected'],ph_missed=outcomes['ph']['missed'],ph_early_alarms=outcomes['ph']['early'],
        ph_median_delay=median('ph'),ddm_fa=float(ddm_fa),ddm_detected=outcomes['ddm']['detected'],
        ddm_missed=outcomes['ddm']['missed'],ddm_early_alarms=outcomes['ddm']['early'],ddm_median_delay=median('ddm'),
        detection_definition='Post-change first alarm only; early alarms and misses are disjoint outcomes',
        threshold_selection='Fixed arguments; no parameter selection on these evaluated streams')


def report(r) -> str:
    return "\n".join([
        "## Exp 58 — CalibrationMonitor: anytime-valid drift detection on the audit stream\n",
        f"In-control loss mean μ0={r['mu0']}, decay to μ1={r['mu1']} at t={r['change_at']}; α={r['alpha']}, "
        f"stream length n={r['n']}. Monitored quantity = per-observation Brier loss (p̂−y)² ∈ [0,1].\n",
        "**1. False-alarm under H0 (pure in-control) — the guarantee the classics lack.**",
        f"| detector | false-alarm rate over {r['n_fa']} streams |",
        "|----------|------------------------------------|",
        f"| **anytime-valid mixture martingale** | **{r['fa_monitor']:.3f}**  (nominal α={r['alpha']:.2f}; finite Monte Carlo estimate) |",
        f"| Page-Hinkley (λ={r['ph_lam']}, δ={r['ph_delta']}) | {r['fa_ph']:.3f}  (fixed threshold for this run, not an α) |",
        f"| naive repeated z-test (no correction) | {r['fa_naive']:.3f}  (inflates — peeking without validity) |",
        "",
        "**2. Detection under injected calibration decay.**",
        f"- **anytime-valid**: detected **{r['detected']}/{r['n_det']}**, early {r['early_alarms']}, missed {r['missed']}; "
        f"**median delay {r['median_delay']:.0f} obs** after the change.",
        f"- **Page-Hinkley** (MLOps handle, λ={r['ph_lam']}/δ={r['ph_delta']} fixed for this run; warm-up {r['ph_min_n']}): detected "
        f"{r['ph_detected']}/{r['n_det']}, early {r['ph_early_alarms']}, missed {r['ph_missed']}; median delay {r['ph_median_delay']:.0f} obs — compare this stream and horizon; the",
        "  difference being that its λ,δ are per-stream tuning knobs while the martingale set α directly.",
        f"- **DDM** (MLOps handle, on its native 0/1 error stream — separate input): FA {r['ddm_fa']:.3f}, detected "
        f"{r['ddm_detected']}/{r['n_det']}, early {r['ddm_early_alarms']}, missed {r['ddm_missed']}, median delay {r['ddm_median_delay']:.0f} obs. Same lesson as PH "
        "(threshold in σ-units, no uniform α); provided for classification-accuracy monitoring.",
        "",
        "Read-out: the mixture betting guarantee requires a fixed valid baseline and a conditional-mean null.",
        "A finite Monte Carlo false-alarm estimate is not the theoretical bound; realized counts can exceed nominal alpha.",
        "PH/DDM are threshold-tuned detectors, not anytime-valid tests. Constant/zero-loss checks alone miss noisy false alarms.",
        "Count first alarms before the injected change as failures, not successful detections; delays use 1-based alarm indices.",
        "Calibrate classic knobs on separate development streams at the intended horizon, scale and dependence; then freeze fresh checks.",
        "ADWIN's adaptive windowing is a complementary policy, not a universally faster detector.",
        "This is the prediction-layer monitor; a fired alarm still hands off to the causal layer (exp38 positivity-",
        "under-shift / exp17 transport) to decide whether the *causal* estimate survived the drift."])


def main():
    import argparse
    p = argparse.ArgumentParser(description="Exp 58: CalibrationMonitor")
    p.add_argument("--n", type=int, default=2000)
    p.add_argument('--ph-delta',type=float,default=.05)
    p.add_argument('--ph-lam',type=float,default=6.)
    p.add_argument('--ph-min-n',type=int,default=1)
    a = p.parse_args()
    r = run(n=a.n,ph_delta=a.ph_delta,ph_lam=a.ph_lam,ph_min_n=a.ph_min_n)
    rep = report(r)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "summary.md").write_text(rep + "\n")
    print(rep)


if __name__ == "__main__":
    main()
