# Three-Level Hierarchical Borrowing Spec (ENCIRCLE SCA)

Regenerated from the June-27 session content and updated with this session's amendments. This is the borrowing-layer specification for the synthetic-control-arm pipeline: how historical/registry information is borrowed across a three-level hierarchy under robust dynamic-borrowing discipline, with the diagnostics and operating-characteristics requirements a CMS/FDA evidence package needs.

## 1. Where borrowing sits in the pipeline

The borrowing layer consumes the TMLE/AIPW patient-level estimand (the 1-year non-hierarchical KM composite of death + HF rehospitalization, tested against the 45% performance goal — per the ENCIRCLE SAP) and borrows strength across a hierarchy to stabilize estimation in rare subgroups. It is **downstream** of the generative augmentation and the causal estimation; it is a Bayesian layer that shares information across strata rather than treating each stratum independently.

**Borrowing scale (this session's amendment).** The KM 1-year composite is a cumulative incidence on [0,1] and is not naturally Normal, so borrowing is performed on the **complementary log-log (cloglog) scale** of the cumulative incidence — the survival-standard, variance-stabilizing link on which the Normal-Normal model of §3 and the size-calibrated cutoff of §5 are well-founded. Concretely, `θ_i = cloglog(F_i)` where `F_i` is the stratum's 1-year composite cumulative incidence, and `s` (§5) is θ_i's influence-function-based SE **on the cloglog scale**. Decisions are reported back on the native rate scale (vs the 45% PG) by inverting the transform.

## 2. The three levels

- **Level 1 — Population.** The overall treatment-effect parameter across all patients.
- **Level 2 — Subgroup.** Pre-specified subgroups. **Amendment (SAP alignment):** ENCIRCLE pre-specifies exactly two subgroups — **subject sex (female vs male)** and **MR etiology (functional vs degenerative)**. The subgroup level of the hierarchy must reconcile to {sex, MR etiology}, or document the deviation at source. This is the load-bearing subgroup-alignment constraint. The subgroup level is the **two marginal factors** (a sex split and an etiology split — four marginal strata), **not** the 2×2 sex×etiology interaction cells: the SAP pre-specifies exactly these two marginal subgroups, and marginal strata avoid the positivity/sparsity failure a crossed cell (e.g. female × degenerative within N=299) would create. If a future analysis targets an interaction cell, it must add explicit positivity/sparsity handling and document the deviation.
- **Level 3 — Patient.** Individual patient-level estimands within subgroups (the rare-subpopulation targets the SCA exists to serve).

Model form (per level, schematically): `y_i | θ_i ~ likelihood`; `θ_i | μ, τ ~ N(μ, τ²)` with `θ_i` on the cloglog scale (§1); `μ ~ weakly-informative Normal`; **`τ ~ half-Normal(0, s₀)`** (Gelman 2006 — preferred over half-Cauchy here because with only 2–4 strata the half-Cauchy's heavy tail can over-borrow; the OC study (§7) characterizes sensitivity to `s₀`). τ (between-group scale) controls shrinkage: small τ → heavy borrowing; large τ → near-independent strata.

## 3. Robust MAP prior (the borrowing mechanism)

Borrowing uses a **robust Meta-Analytic-Predictive (MAP) prior** (Schmidli et al. 2014):

```
π_rMAP(θ) = w · π_MAP(θ) + (1 − w) · π_vague(θ)
```

- `π_MAP` is the informative component (the borrowed prior).
- `π_vague` is a weakly-informative escape component.
- `w` is the mixture weight on the **informative** component; `(1−w)` is the weight on the **vague** component. (Convention guard below.)
- Dynamic borrowing property: concordant data → posterior dominated by the MAP component (borrow); discordant data → the vague component takes over and historical information is automatically discounted. No separate conflict-detection rule is required — the robust mixture handles prior–data conflict by construction.

**w-convention guard (this session's correction).** In the RBesT `robustify` convention, the robust weight is the weight on the **vague** component (higher robust weight = *less* borrowing). Whichever convention the kernel uses must be verified in code, because the direction of every w-dependent result inverts under the opposite convention. The flip-behavior test (below) depends on this.

Power prior, commensurate prior, and robust mixtures are all special cases of BHMs (Yang et al. 2023); the three-level hierarchy here is the general machinery, robust-MAP the chosen parameterization at each level.

## 4. Effective sample size — three distinct notions (do NOT cross-wire)

This session's terminology guard, load-bearing across the whole pipeline:

- **Prior-ESS (this layer).** Morita, Thall & Müller (2008) variance-ratio ESS — "how many virtual patients is the informative prior worth." Analytical, no bulk/tail decomposition. Report at the posterior median of τ and at sensitivity points. This is the ESS the borrowing layer reports. **(Confirmed already implemented via `compute_ess`, Morita-Thall-Müller variance-ratio — do not change.)**
- **Particle-ESS (SMC layer, not this layer).** Kish (Σw)²/Σw² on particle weights; resampling trigger. Different object.
- **MCMC bulk/tail-ESS (OC fidelity fits only, not this layer).** Autocorrelation-based; applies only where a non-conjugate model is fit by MCMC (see §7). Tail-ESS is load-bearing there because Type I error is a tail event.

The borrowing kernel is **analytically tractable** — the robust mixture posterior is a mixture of conjugate component posteriors (closed-form, but a *mixture*, not conjugate in the single-component sense; the mixture-weight update is exactly what §8's influence factor measures) — and needs **no** bulk/tail-ESS. Only the non-conjugate OC fits (§7) do.

## 5. Size-calibrated decision cutoff (NESS slides 45–48 — highest-priority refinement)

Under an informative prior, holding the success cutoff at the vague-prior value (z = 1.96 / c = 0.975) lets the implied one-sided Type I error drift up (~10% in the slides' counter-example). Use the **size-calibrated cutoff**:

```
c* = Φ(b + k · z_{1−α}),   k = r / sqrt(1 + r²),   r = τ / s
```

where τ is the prior SD and s is the likelihood SE **of the cloglog-scale stratum estimand (§1)** — the influence-function-based SE of `cloglog(F_i)`, on the same scale as the borrowing itself. As r → ∞ (vague), c* → 0.975 (the fixed cutoff is the vague-prior limit only). For finite r (the borrowing regime), calibrate. Report `r = τ/s` (or prior-ESS) alongside the cutoff. **This is exact for the Normal-Normal population level; for patient-level / non-Normal paths use the local-approximation treatment (Hansen & Tong conjugacy-deviation bound).**

## 6. Tipping-point / sensitivity discipline

The borrowing conclusion must be reported as a function of how much is borrowed and how much conflict is present:

- **Borrowing-strength tipping point (τ axis).** Sweep `tau_prior_sd` × conflict as a 2D surface; report where the conclusion flips. (Implemented: exp20.)
- **Mixture-weight tipping point (w axis) — refinement.** The NESS submission checklist (slide 230) names a "tipping-point sweep over the **borrowing weight**" — the mixture weight w, ranged over [0.1, 0.9], Brensocatib-style. Add w as a swept axis alongside τ (regular w-grid per (τ, conflict) cell, report min-flip-w; no QMC, no spline — the model is analytical, a regular grid finds the flip directly and legibly).
- **Flip-direction test (this session's correction).** Under conflict, the informative prior props up a conclusion the data doesn't support, so only a small robust_weight is needed to flip it → `flip_robust_weight` is **lower** under conflict than under concordance; concordant data may never flip (NaN). Test: `flip_robust_weight(conflict) < flip_robust_weight(concordance)`, allow NaN on the concordant side, detect flip as first crossing, verify the w-convention in code.
- **Worst-case event-imputation tipping point (SAP alignment).** ENCIRCLE pre-specifies a censoring/missing-data tipping point (Figure S4): sweep j = 0…k censored subjects converted event-free → event, recompute the PG test, report j*/k at the critical-value crossing. This is a distinct, missing-data tipping point (maps to exp13), complementary to the borrowing-strength and confounding tipping points.

## 7. Operating characteristics and two-vs-three-level fidelity

- **Frequentist translation (NESS through-line).** Report Type I error, power, coverage — plus Type-M/Type-S (Gelman-Carlin) and MDE. The OC study is where the borrowing design is validated against regulatory quantities.
- **Conjugacy regime.** The two-level Normal-Normal robust-MAP is **analytically tractable** — a closed-form mixture of conjugate component posteriors (see §4), fast; the OC inner loop runs analytically at present. The **full three-level BHM** (the `half-Normal(0, s₀)` hyperprior on τ) is genuinely **non-conjugate** and requires MCMC.
- **Two-vs-three-level fidelity (this session's amendment).** Rather than leaning on the mixture-linearization / conjugacy-deviation diagnostic (Hansen & Tong, issue 16) to argue the two-level conjugate OC approximates the three-level OC, fit the full three-level BHM via **MCMC (NumPyro/JAX, GPU-parallel on the A100s) on a subsample of replicates (200–500)** and compare the OC quantities (Type I, power, coverage, decision reversals) directly against the conjugate kernel on the *same* replicates. Retain the deviation diagnostic as a complementary analytical bound.
- **Tail-ESS on the MCMC fits (required).** Each three-level MCMC fit reports R-hat, bulk-ESS, and **tail-ESS**. Tail-ESS is load-bearing because Type I error is a tail event and the estimand concerns the rare subpopulation. Flag fits below threshold; do not silently average them in.

*This section and §6 are operationalized end-to-end in `oc_simulation_pipeline.mermaid` (the OC simulation pipeline diagram).*

## 8. Prior–data conflict diagnostics

- **Robust-mixture escape (primary).** The vague component is the built-in conflict handler; overlay prior and likelihood — if they barely overlap, "the posterior is borrowing the disagreement, not resolving it."
- **Influence factor (IF) — named diagnostic (NESS slide 221/230).** `log IF = log Pr_M(A|y) − log Pr_V(A|y)` at the decision threshold (M = informative/MAP component, V = vague). |log IF| small → agreement; large → the informative component pulls against the data. Add alongside the existing `approximation_deviation` (which measures the same phenomenon but is not the named IF).
- **Reversal probability.** Design-stage probability that the borrowing decision reverses relative to the no-borrowing (vague-only) decision, over the conflict grid — the slides' "reversal probability surface."

## 9. Anti-circularity and layer discipline

- Synthetic rare patients must never be folded back into training data at the encoder or diffusion layer; borrowing operates on real-patient estimands. The specific failure this prevents: folding synthetic patients back into training makes the generative prior and the borrowing estimand **share information**, breaking the independence the inferential layer assumes and **inflating apparent precision by double-counting the same underlying evidence** — a silent Type I error inflation, not merely a tidiness rule.
- The borrowing layer is inferential; it must not import ranking metrics or generative-layer quantities.
- Measurement-error (embedding uncertainty Σ_ε) enters as a *sensitivity analysis* consuming a supplied Σ_ε (SIMEX / error-in-variables), not as a change to the primary borrowing estimand.

## References
- Schmidli et al. (2014) — robust MAP prior.
- Morita, Thall & Müller (2008) — prior effective sample size.
- Yang et al. (2023) — BHM unification of borrowing methods.
- Hansen & Tong (2026) — conjugacy-deviation diagnostic (issue 16).
- Vehtari et al. (2021) — bulk-ESS / tail-ESS (MCMC fits only).
- ENCIRCLE SAP (Lancet suppl., Section C) — endpoint, subgroups, tipping point.
- NESS 2026 "Bayesics" short course — size calibration (45–48), IF (221), submission checklist (230).
