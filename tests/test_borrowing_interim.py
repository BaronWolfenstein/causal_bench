"""exp47 (#196): borrowing × interim-looks composition."""
from causal_bench.validation.borrowing_interim import composition_type_i


def _grid(rows):
    return {(r["policy"], r["delta"], r["method"]): r["type_i"] for r in rows}


def test_no_borrow_multiplicity_controlled_by_obf_and_cs():
    """flat (no borrow): naive multiple-looks inflates Type-I above α, but OBF and the confidence
    sequence control it — sequential.py works for a non-borrowing statistic."""
    g = _grid(composition_type_i([0.0], ("flat",), K=5, n_reps=1500, seed=1))
    assert g[("flat", 0.0, "naive")] > 0.07          # multiplicity inflation
    assert g[("flat", 0.0, "obf")] <= 0.08           # OBF controls ~α
    assert g[("flat", 0.0, "confidence_sequence")] <= 0.05


def test_borrowing_under_conflict_breaks_all_monitors():
    """map under conflict (δ>0): borrowing centers the statistic at δ, so cumulative Type-I blows
    up under EVERY monitoring rule — the composition inflation neither component test reveals."""
    g = _grid(composition_type_i([0.30], ("map",), K=5, n_reps=1200, seed=2))
    assert g[("map", 0.30, "naive")] > 0.5
    assert g[("map", 0.30, "obf")] > 0.5             # OBF does NOT rescue borrowing-conflict
    assert g[("map", 0.30, "confidence_sequence")] > 0.3


def test_no_conflict_borrow_is_not_inflated():
    """δ=0 (historical also null): borrowing a null prior does not inflate (if anything it's
    conservative) — conflict, not borrowing per se, is the hazard."""
    g = _grid(composition_type_i([0.0], ("map",), K=5, n_reps=1200, seed=3))
    assert g[("map", 0.0, "obf")] <= 0.08
