# Zero-Flow CI — Torch (neural-velocity) Extension Spec

**Status:** deferred / spec-only. The CPU numpy-sklearn estimator (`causal_bench/detectors/zero_flow_ci.py`, PR #92 / issue #85) is the default and is sufficient for tabular, low-dimensional CI testing. This spec documents the **escape hatch** — a torch neural-velocity backend — so the extension path is fixed before merge. Do not build until a warranting regime (below) actually appears.

## Why an extension at all (the boundary)

The CPU estimator computes the zero-flow statistic as the rectified-flow velocity at t=0.5, `E[x1−x0 | x_0.5]`, via a **cross-fitted regressor** (default `LinearRegression` for the velocity; RF for residualizing on Z). Two facts bound when that's enough:

- The zero-flow criterion only needs the velocity **at the single time-slice t=0.5**, and the flow-matching optimum there *is* that conditional expectation — so we never train a full flow; a regression suffices. Torch's main asset (training an expressive flow over all t) is **orthogonal to the test**.
- `Cov(v, x_0.5) = ½(Σ₁ − Σ₀)`, so a *linear* velocity already detects covariance/correlation-type CI violations; a nonlinear sklearn velocity (RF/GB, pluggable via `velocity_factory`) covers moderate nonlinear dependence with no GPU.

**A torch backend is warranted only when BOTH of these bite:**
1. **High-dimensional** X / Y / Z — e.g., conditional independence between *learned embeddings* (patient-trajectory embeddings, hundreds of dims), where sklearn RF/linear velocities scale and generalize poorly; and
2. **large n** — enough data that a neural velocity's variance is controlled.

It is **not** warranted for tabular low-dim CI (the sklearn path is simpler, faster, CPU, and links cleanly into SGA), and it does **not** fix Type-I calibration (that's driven by residualization quality + the permutation null — a separate axis; `n_perm` only sets the p-value's Monte-Carlo precision, not the test's size).

## Interface — drops into the existing hook (no API break)

The estimator already exposes `velocity_factory` (and `nuisance_factory`). The torch backend is a fit/predict object satisfying the same contract, so `zero_flow_ci_test(..., velocity_factory=TorchVelocity)` needs **zero change** to the test logic, cross-fitting, residualization, or the permutation calibration.

```python
# causal_bench/detectors/zero_flow_torch.py   (deferred)
class TorchVelocity:
    """Small MLP velocity v_θ(x_0.5) -> (x1 - x0), sklearn-style fit/predict.
    Single time-slice (t=0.5) — NOT a full-flow trainer; the criterion needs
    only that slice. Lazy torch import; device via resolve_device."""
    def __init__(self, hidden=256, epochs=200, lr=1e-3, device="auto"): ...
    def fit(self, X, V): ...      # X=(n,d) x_0.5, V=(n,d) target velocity
    def predict(self, X): ...     # returns numpy (n,d); test stays device-agnostic
```

`CITestResult` is unchanged, so **SGA linkage is unaffected** — the empirical-leg verdict still maps 1:1 onto `EmpiricalCIResult`, and torch never enters SGA (the estimator runs in causal_bench; SGA consumes the verdict).

## Tasks (when built)

1. **`[flow]` optional extra** in `pyproject.toml` (`torch>=2.2`); lazy-imported inside `zero_flow_torch.py` only. Mirrors the A100 deployment spec's dependency discipline (never torch at module top level).
2. **`TorchVelocity`** — MLP, MSE flow-matching loss on the t=0.5 slice, `resolve_device` (cuda→mps→cpu), bf16 autocast on CUDA (Tensor Cores), `predict` returns host numpy. `importorskip("torch")` tests.
3. **Parity test (CPU-torch, small net):** on the tabular DGPs from `test_zero_flow_ci.py`, `velocity_factory=TorchVelocity` must reproduce the sklearn verdicts (independent→supports, dependent→refutes).
4. **Power test (the reason it exists):** a **high-dim, nonlinear** DGP where the linear velocity fails to refute a true dependence but `TorchVelocity` refutes it — the concrete evidence the extension earns its keep.
5. **Optional torch nuisance** for residualizing high-dim Z (same `nuisance_factory` hook).

## Scale / GPU

Per the A100 deployment spec: GPU only for the high-dim × large-n regime; the net is tiny (checkpoints trivial); CPU-torch validates correctness, the A100 validates throughput. The permutation calibration (n_perm × cross-fit) is the cost driver — on GPU, batch the per-permutation velocity fits.

## Decision record

- Default stays **numpy/sklearn** (this PR). Torch is opt-in via `velocity_factory`.
- Trigger to build: a real **embedding-space CI** need (e.g. testing independence between patient-trajectory embeddings — ties to the diffuse_directly / localization embedding work, and to SGA only if claims ever acquire embedding-derived data).
- Non-triggers: tabular CI, nonlinear-but-low-dim (use an sklearn RF/GB velocity), or Type-I calibration concerns (fix via residualization quality / the permutation null and more data — `n_perm` only sharpens MC precision, not size).
