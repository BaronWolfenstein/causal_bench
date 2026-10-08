"""exp58 — CalibrationMonitor: an anytime-valid drift detector for a production calibration/accuracy stream.

The rigorous core of the drift stack (docs/plans/2026-09-24-drift-validity-monitoring-stack.md). Montgomery's
calibration loop refits periodically and validates on held-back data; this replaces the fixed-cadence check with
a **mixture betting test-martingale** (Waudby-Smith–Ramdas / Howard-style e-process) over the labeled audit stream:
you may peek after *every* observation and refit on the first alarm, with a uniform false-alarm guarantee.

Monitored quantity: a per-observation bounded loss ℓ_t ∈ [0,1] on the audit sample — here the Brier component
(p̂−y)² — whose in-control mean is μ0 (the calibrated baseline). Calibration/accuracy DECAY raises E[ℓ], which the
detector catches.

Construction (H0: E[ℓ_t | past] ≤ μ0 with a valid, fixed baseline). Per-step e-value eₜ(λ) = 1 + λ(ℓₜ − μ0) with λ ∈ [0, 1/μ0) (keeps eₜ ≥ 0 for
ℓₜ ∈ [0,1]); wealth Kₜ(λ) = ∏ eₛ(λ) is a test (super)martingale under H0, and the **mixture** K̄ₜ = mean_λ Kₜ(λ)
is too. **Ville's inequality ⇒ P(sup_t K̄ₜ ≥ 1/α | H0) ≤ α** — a uniform, anytime-valid false-alarm bound. Alarm
the first time K̄ₜ ≥ 1/α. During in-control K̄ₜ stays O(1) (no growth); after an upward shift it grows
exponentially, with CUSUM-like detection; relative delay depends on the stream and tuning, rather than
universally dominating Page-Hinkley/DDM. The false-alarm guarantee requires the conditional-mean null. A sliding/adaptive window (ADWIN-style
forgetting) further sharpens LATE-change delay and is the natural companion (the "when to forget/refit" policy);
it is complementary, not subsumed.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
import math
from numbers import Real
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
@dataclass
class StreamingPageHinkley:
    """Scale-dependent first-alarm monitor with JSON-restorable state.

    Calibration must use separate development streams and the intended monitoring
    horizon, noise scale and dependence. Neither lam nor min_n is an alpha level.
    Real-valued finite scalars are accepted; arrays and binary booleans are rejected.
    """
    delta: float = 0.005
    lam: float = 1.0
    min_n: int = 1
    n: int = 0
    mean: float = 0.0
    cumulative: float = 0.0
    minimum: float = 0.0
    alarm: bool = False
    detected_at: int | None = None

    def __post_init__(self):
        for name in ('delta','lam','mean','cumulative','minimum'):
            value=getattr(self,name)
            if isinstance(value,(bool,np.bool_)) or not isinstance(value,Real) or not math.isfinite(value):
                raise ValueError('Finite scalar '+name+' required')
            setattr(self,name,float(value))
        if self.delta<0 or self.lam<=0 or type(self.min_n) is not int or self.min_n<1:
            raise ValueError('Nonnegative slack, positive threshold and integer warm-up required')
        if type(self.n) is not int or self.n<0 or type(self.alarm) is not bool or self.minimum>self.cumulative:
            raise ValueError('Malformed Page-Hinkley state')
        if (self.alarm!=(self.detected_at is not None) or (self.detected_at is not None and
            (type(self.detected_at) is not int or not self.min_n<=self.detected_at<=self.n))):
            raise ValueError('First alarm index/state inconsistent')
        if self.n==0 and any(getattr(self,k)!=0 for k in ('mean','cumulative','minimum')):
            raise ValueError('Empty monitor has nonempty statistics')

    def update(self,value):
        if isinstance(value,(bool,np.bool_)) or not isinstance(value,Real) or not math.isfinite(value):
            raise ValueError('Page-Hinkley requires a finite real scalar')
        x=float(value)
        n=self.n+1
        mean=self.mean+(x-self.mean)/n
        cumulative=self.cumulative+(x-mean-self.delta)
        if not math.isfinite(mean) or not math.isfinite(cumulative):
            raise ValueError('Page-Hinkley statistics overflowed; state unchanged')
        self.n=n;self.mean=mean;self.cumulative=cumulative
        self.minimum=min(self.minimum,self.cumulative)
        if not self.alarm and self.n>=self.min_n and self.cumulative-self.minimum>self.lam:
            self.alarm=True;self.detected_at=self.n
        return self.alarm

    def state_dict(self):
        return asdict(self)

    @classmethod
    def from_state_dict(cls,state):
        if not isinstance(state,dict) or set(state)!=set(cls.__dataclass_fields__):
            raise ValueError('Page-Hinkley state fields differ')
        return cls(**state)


def page_hinkley(stream, *, delta: float = 0.005, lam: float = 1.0, min_n: int = 1):
    """First 1-based alarm on an upward-mean change, else None.

    Default min_n=1 preserves historical batch behavior for finite real inputs.
    Warm-up prevents alarms on the initial min_n-1 observations, but does not
    calibrate the finite-horizon false-alarm rate. Tune
    slack/threshold on independent noisy development streams; count pre-change
    alarms separately from successful drift detections. A threshold is not alpha.
    """
    monitor=StreamingPageHinkley(delta=delta,lam=lam,min_n=min_n)
    for value in stream:
        if monitor.update(value):return monitor.detected_at
    return None


def ddm(errors, *, min_n: int = 30, warn_k: float = 2.0, drift_k: float = 3.0):
    """DDM (Gama et al. 2004) for a 0/1 ERROR stream: warning at p+s > p_min+warn_k·s_min, drift at drift_k·s_min,
    with p the running error rate and s=√(p(1−p)/n). Threshold-based (k in std units), no uniform α. Returns
    (warning_at, drift_at).

    Note the STRICT inequalities: drift means the current error band must *exceed* the historical-minimum band,
    not merely equal it. With a non-strict `>=` a degenerate zero-variance baseline false-alarms — e.g. a clean
    `ddm([0]*100)` stream has p=sd=0 throughout, so after the band is set to p_min=s_min=0 the drift test
    `0 >= 0+3·0` would fire spuriously at the first checked step. `>` makes "no increase" correctly a non-alarm."""
    n = 0; s = 0.0; p_min = float("inf"); s_min = float("inf"); warn = None
    for t, e in enumerate(errors, 1):
        n += 1; s += float(e); p = s / n; sd = (p * (1 - p) / n) ** 0.5
        if n < min_n:
            continue
        if p + sd <= p_min + s_min:
            p_min = p; s_min = sd
        if p + sd > p_min + drift_k * s_min:
            return (warn, t)
        if warn is None and p + sd > p_min + warn_k * s_min:
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
