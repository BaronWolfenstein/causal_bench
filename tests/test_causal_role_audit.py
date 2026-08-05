"""Self-validating checks for the latent-aware collider audit (issue #216)."""
import pytest

pytest.importorskip("lingam")  # RCD + ParceLiNGAM; skip where the optional dep is absent

from causal_bench.validation.causal_role_audit import audit, lingam_ensemble, sim_latent_collider


def test_latent_collider_audit_excludes_the_collider():
    """On an M-bias collider with HIDDEN parents (Ha->A, Hy->Y, Cm<-Ha,Hy), the ensemble audit
    flags Cm for exclusion — the collider PC/orient_colliders and constraint-FCI cannot orient."""
    X, names = sim_latent_collider(n=5000, seed=0)
    r = audit(X, names)
    assert "Cm" in r["exclude_covariates"]          # the latent collider must be excluded
    assert "Cm" not in r["safe_adjustment_set"]


def test_ensemble_is_more_complete_than_rcd_alone():
    """RCD has false negatives; ParceLiNGAM covers a pair RCD misses (verified: Cm<->Y). The
    conservative union catches both true latent pairs — the reason to ensemble, not trust one."""
    X, names = sim_latent_collider(n=5000, seed=0)
    ens = lingam_ensemble(X, names)
    assert frozenset(("A", "Cm")) in ens["rcd"]                 # RCD catches A<->Cm
    assert frozenset(("Cm", "Y")) in ens["parcelingam"]         # ParceLiNGAM adds Cm<->Y (RCD missed)
    assert frozenset(("Cm", "Y")) in ens["union"]              # union has both true latent pairs
    assert frozenset(("A", "Cm")) in ens["union"]
