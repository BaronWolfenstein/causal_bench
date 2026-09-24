"""exp58 — CalibrationMonitor: an anytime-valid drift detector for a production calibration/accuracy stream.

The rigorous core of the drift stack (docs/plans/2026-09-24-drift-validity-monitoring-stack.md). Montgomery's
calibration loop refits periodically and validates on held-back data; this replaces the fixed-cadence check with
a **mixture betting test-martingale** (Waudby-Smith–Ramdas / Howard-style e-process) over the labeled audit stream:
you may peek after *every* observation and refit on the first alarm, with a uniform false-alarm guarantee.

Monitored quantity: a per-observation bounded loss ℓ_t ∈ [0,1] on the audit sample — here the Brier component
(p̂−y)² — whose in-control mean is μ0 (the calibrated baseline). Calibration/accuracy DECAY raises E[ℓ], which the
detector catches.

Construction (H0: E[ℓ] ≤ μ0). Per-step e-value eₜ(λ) = 1 + λ(ℓₜ − μ0) with λ ∈ [0, 1/μ0) (keeps eₜ ≥ 0 for
ℓₜ ∈ [0,1]); wealth Kₜ(λ) = ∏ eₛ(λ) is a test (super)martingale under H0, and the **mixture** K̄ₜ = mean_λ Kₜ(λ)
is too. **Ville's inequality ⇒ P(sup_t K̄ₜ ≥ 1/α | H0) ≤ α** — a uniform, anytime-valid false-alarm bound. Alarm
the first time K̄ₜ ≥ 1/α. During in-control K̄ₜ stays O(1) (no growth); after an upward shift it grows
exponentially, so detection delay ≈ log(1/α)/KL — CUSUM-like, hence it subsumes Page-Hinkley/DDM on *detection*
too, while (unlike them) controlling the false-alarm rate uniformly. A sliding/adaptive window (ADWIN-style
forgetting) further sharpens LATE-change delay and is the natural companion (the "when to forget/refit" policy);
it is complementary, not subsumed.
"""
from __future__ import annotations
import numpy as np


class CalibrationMonitor:
    """Streaming anytime-valid drift monitor. `update(loss)` per audit observation; `.alarm` latches True the first
    time the mixture wealth crosses 1/α, and `.detected_at` records that index. `mu0` = in-control loss mean."""

    def __init__(self, mu0: float, *, alpha: float = 0.05, n_lambda: int = 20, lam_frac: float = 0.5):
        self.mu0 = float(mu0)
        self.alpha = float(alpha)
        self.threshold = 1.0 / alpha
        # bet grid in [0, lam_frac/μ0) — lam_frac<1 keeps eₜ safely positive and trades a little power for stability
        self.lams = np.linspace(0.0, lam_frac / max(mu0, 1e-6), n_lambda)
        self.K = np.ones(n_lambda)            # per-λ wealth
        self.t = 0
        self.wealth_hist: list[float] = []
        self.alarm = False
        self.detected_at: int | None = None

    def update(self, loss: float) -> float:
        self.t += 1
        self.K = self.K * (1.0 + self.lams * (float(loss) - self.mu0))
        self.K = np.clip(self.K, 0.0, None)   # bounded loss ⇒ factor ≥ 0; clip guards float underflow
        w = float(self.K.mean())              # mixture wealth
        self.wealth_hist.append(w)
        if not self.alarm and w >= self.threshold:
            self.alarm = True
            self.detected_at = self.t
        return w


def run_monitor(losses, mu0, *, alpha=0.05, **kw):
    """Batch convenience: feed a loss stream, return (wealth_history, detected_at | None)."""
    m = CalibrationMonitor(mu0, alpha=alpha, **kw)
    for x in losses:
        m.update(x)
    return np.asarray(m.wealth_hist), m.detected_at


def naive_repeated_test(losses, mu0, *, alpha=0.05):
    """Baseline that INFLATES: a fresh one-sided z-test of the running mean > μ0 at every step, alarming on the
    first p<alpha. No multiplicity correction ⇒ its false-alarm rate under H0 far exceeds alpha (the thing the
    anytime-valid monitor fixes)."""
    from scipy import stats
    s = 0.0; ss = 0.0
    for t, x in enumerate(losses, 1):
        s += x; ss += x * x
        if t < 5:
            continue
        mean = s / t
        var = max(ss / t - mean ** 2, 1e-9)
        z = (mean - mu0) / np.sqrt(var / t)
        if 1.0 - stats.norm.cdf(z) < alpha:
            return t
    return None


# ---------------------------------------------------------------- classic MLOps handles (familiar, threshold-tuned)
# Provided for interop/familiarity: DDM and Page-Hinkley are what most MLOps drift tooling exposes. They DETECT
# well but are THRESHOLD-tuned — there is no uniform P(false alarm) ≤ α guarantee (unlike the anytime-valid
# CalibrationMonitor); their knobs are scale-dependent and must be tuned per stream. Use them as familiar
# front-ends / baselines; use CalibrationMonitor when you need a principled false-alarm level.
def page_hinkley(stream, *, delta: float = 0.005, lam: float = 1.0):
    """Page-Hinkley CUSUM detector for an INCREASE in the mean of a real-valued stream. `delta` = slack (drift
    magnitude to ignore), `lam` = detection threshold (a tuned knob, NOT a false-alarm level). Returns the first
    alarm index, else None."""
    x_mean = 0.0; m = 0.0; M = 0.0
    for t, x in enumerate(stream, 1):
        x_mean += (float(x) - x_mean) / t
        m += (float(x) - x_mean - delta)
        M = min(M, m)
        if m - M > lam:
            return t
    return None


def ddm(errors, *, min_n: int = 30, warn_k: float = 2.0, drift_k: float = 3.0):
    """DDM (Gama et al. 2004) for a 0/1 ERROR stream: warning at p+s ≥ p_min+warn_k·s_min, drift at drift_k·s_min,
    with p the running error rate and s=√(p(1−p)/n). Threshold-based (k in std units), no uniform α. Returns
    (warning_at, drift_at)."""
    n = 0; s = 0.0; p_min = float("inf"); s_min = float("inf"); warn = None
    for t, e in enumerate(errors, 1):
        n += 1; s += float(e); p = s / n; sd = (p * (1 - p) / n) ** 0.5
        if n < min_n:
            continue
        if p + sd <= p_min + s_min:
            p_min = p; s_min = sd
        if p + sd >= p_min + drift_k * s_min:
            return (warn, t)
        if warn is None and p + sd >= p_min + warn_k * s_min:
            warn = t
    return (warn, None)


def simulate_loss_stream(n, mu0, *, mu1=None, change_at=None, seed=0):
    """A [0,1] Brier-like loss stream: Beta with mean μ0 (in-control), optionally shifting to mean μ1 at
    `change_at`. `mu1=None` ⇒ pure in-control (for false-alarm calibration)."""
    rng = np.random.default_rng(seed)
    conc = 8.0
    m = np.full(n, mu0, float)
    if mu1 is not None and change_at is not None:
        m[change_at:] = mu1
    return rng.beta(m * conc, (1 - m) * conc)


def simulate_error_stream(n, p0, *, p1=None, change_at=None, seed=0):
    """A 0/1 ERROR stream (DDM's native input): Bernoulli(p0) in-control, optionally shifting to Bernoulli(p1)
    at `change_at`."""
    rng = np.random.default_rng(seed)
    p = np.full(n, p0, float)
    if p1 is not None and change_at is not None:
        p[change_at:] = p1
    return (rng.random(n) < p).astype(int)
