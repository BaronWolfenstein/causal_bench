"""exp44 Part-B — the residual-borrowing LAYER (general, benchmark version).

Composes the pieces built in Part A into the gated, PS-integrated borrow:

    gate (residual_conflict_gate)  →  protocol response (#180)  →  PS-integrated robust-MAP.

Part A already supplies the pseudo-study meta (`sim_epoch` with n_sub = a set of pseudo-studies)
and the robust-MAP fit (`_fit_mu` / `map_prior_from_historical`). This layer adds the two missing
generic pieces: **PS-integration** (down-weight historical pseudo-studies by their propensity
comparability to the current population) and the **protocol response** (the prespecified action
keyed to the gate verdict). The ENCIRCLE-specific instantiation — real TVT partitions + the
frozen-embedding propensity — is NOT here; this is the reusable, simulated-data version.

Protocol response (#180), keyed to the gate:
  gate 'supports'      (no residual conflict) → `map`         — borrow fully; exchangeable.
  gate 'refutes'       (residual conflict)    → `robust_map`  — borrow with the vague-mixture
                                                                 safety; down-weights under conflict.
  gate 'underpowered'  (can't certify)        → `flat`        — do not borrow; conservative.
Each response is a different composed design, so its operating characteristics must be read from
Part A's Type-I(δ) sweep for the policy that fired.
"""
from __future__ import annotations

import numpy as np

from causal_bench.validation.residual_conflict_gate import residual_conflict_gate
from causal_bench.validation.two_epoch_borrowing import _fit_mu, map_prior_from_historical


def ps_integrated_map_prior(theta_h, se_h, ps_weight, *, tau_sd=0.5, draws=500, tune=500,
                            chains=2, seed=0):
    """PS-integrated MAP prior. `ps_weight` ∈ (0,1] is each historical pseudo-study's propensity
    comparability to the current population (1 = fully comparable). Less-comparable studies are
    down-weighted by inflating their se (se_eff = se / √ps_weight), so their Fisher information —
    and thus their pull on the borrowed prior — shrinks. Returns (map_mean, map_sd)."""
    w = np.clip(np.asarray(ps_weight, float), 1e-3, 1.0)
    se_eff = np.asarray(se_h, float) / np.sqrt(w)
    return map_prior_from_historical(theta_h, se_eff, tau_sd=tau_sd, draws=draws, tune=tune,
                                     chains=chains, seed=seed)


def protocol_response(gate_result: dict) -> str:
    """#180 prespecified response keyed to the gate verdict → the policy that fires."""
    if gate_result.get("residual_conflict") is None:      # underpowered
        return "flat"
    return "robust_map" if gate_result["residual_conflict"] else "map"


def gated_borrow(current_theta, current_se, hist_theta, hist_se, ps_weight, *,
                 gate_outcome, gate_source, gate_embedding, w_vague=0.5, vague_sd=1.0,
                 tau_sd=0.5, draws=500, tune=500, chains=2, seed=0, gate_n_perm=100):
    """The full layer: run the residual-conflict gate, pick the protocol response, form the
    PS-integrated prior, and fit the current epoch. Returns the decision + which policy fired.

    `gate_outcome/source/embedding` feed the gate (a control/negative-control outcome, source
    indicator, pre-treatment embedding). `hist_theta/se` + `ps_weight` are the historical
    pseudo-studies; `current_theta/se` the current epoch."""
    gate = residual_conflict_gate(gate_outcome, gate_source, gate_embedding,
                                  n_perm=gate_n_perm, rng=np.random.default_rng(seed))
    policy = protocol_response(gate)
    if policy == "flat":
        prior = ("normal", 0.0, vague_sd)
    else:
        m, s = ps_integrated_map_prior(hist_theta, hist_se, ps_weight, tau_sd=tau_sd,
                                       draws=draws, tune=tune, chains=chains, seed=seed)
        prior = ("normal", m, s) if policy == "map" else ("robust", w_vague, vague_sd, m, s)
    fit = _fit_mu(current_theta, current_se, mu_prior=prior, tau_sd=tau_sd, draws=draws,
                  tune=tune, chains=chains, seed=seed)
    return {"policy_fired": policy, "gate_verdict": gate["verdict"],
            "residual_conflict": gate.get("residual_conflict"),
            "rejects_null": fit["rejects_null"], "effect": fit["effect"],
            "ci_lo": fit["ci_lo"], "ci_hi": fit["ci_hi"]}
