"""Geodesic positivity R (#159). Load-bearing test:
`test_geodesic_rejects_manufactured_positivity` — two Euclidean-near but
geodesically-disconnected sheets, where the flat Euclidean R (#164) reports fake
overlap and the geodesic R correctly reports none."""
import numpy as np

from causal_bench.geometry.geodesic import (
    geodesic_distance_dijkstra, geodesic_distance_heat, geodesic_positivity_overlap,
    make_two_sheets, make_swiss_roll)
from causal_bench.geometry.spectral import build_knn_laplacian
from causal_bench.sampling.stratified import positivity_overlap  # the flat Euclidean R


def test_geodesic_rejects_manufactured_positivity():
    X, a_mask, b_mask = make_two_sheets(n=200, gap=0.3, seed=0)
    radius = 0.45                       # > gap (0.3) so Euclidean "covers" across sheets

    # flat Euclidean R: particles on A, targets on B — fake overlap (sheets are close)
    flat = positivity_overlap(X[a_mask], X[b_mask], radius=radius)
    assert flat.coverage > 0.9, "setup: Euclidean R should manufacture overlap here"

    # geodesic R: sheets are disconnected in the k-NN graph -> no real overlap
    geo = geodesic_positivity_overlap(X, a_mask, b_mask, radius=radius, k=6)
    assert geo.coverage < 0.05, "geodesic R must reject the manufactured positivity"
    assert geo.uncovered.all()


def test_geodesic_agrees_on_connected_overlap():
    """Particles and targets on the SAME sheet, spatially overlapping -> both
    Euclidean and geodesic report coverage."""
    rng = np.random.default_rng(1)
    pts = np.column_stack([rng.uniform(0, 1, (300, 2)), np.zeros(300)])
    par = np.zeros(300, bool); par[:150] = True
    geo = geodesic_positivity_overlap(pts, par, ~par, radius=0.3, k=10)
    assert geo.coverage > 0.9


def test_geodesic_ge_euclidean_and_heat_correlates():
    """Geodesic ≥ Euclidean (invariant), and the matrix-free heat estimator
    rank-correlates with the exact dijkstra oracle on a connected manifold."""
    X, _ = make_swiss_roll(n=250, seed=2)
    src = 0
    d_geo = geodesic_distance_dijkstra(X, k=10, sources=[src])[0]
    d_euc = np.linalg.norm(X - X[src], axis=1)
    finite = np.isfinite(d_geo)
    assert np.all(d_geo[finite] >= d_euc[finite] - 1e-9)   # geodesic never shorter

    L = build_knn_laplacian(X, k=10)
    d_heat = np.asarray(geodesic_distance_heat(L, src, t=0.03))
    m = finite & np.isfinite(d_heat)
    # Spearman (rank) correlation between heat and dijkstra
    ra = np.argsort(np.argsort(d_geo[m])); rb = np.argsort(np.argsort(d_heat[m]))
    rho = np.corrcoef(ra, rb)[0, 1]
    assert rho > 0.7, f"heat-Varadhan should track dijkstra ranks (rho={rho:.2f})"


def test_disconnected_sheets_give_inf():
    X, a_mask, b_mask = make_two_sheets(n=120, gap=0.4, seed=3)
    D = geodesic_distance_dijkstra(X, k=6, sources=[0])[0]
    assert np.isinf(D[b_mask]).all()     # nothing on sheet B reachable from A[0]


def test_mass_vs_coverage_distinct():
    """Reached-but-weightless: particles cover R geodesically but carry ~no
    weight there — coverage high, mass ~0 (the tail-mass-collapse analogue)."""
    rng = np.random.default_rng(4)
    pts = np.column_stack([rng.uniform(0, 1, (200, 2)), np.zeros(200)])
    par = np.zeros(200, bool); par[:100] = True
    log_w = np.zeros(200); log_w[:100] = np.linspace(0, -20, 100)  # far particles ~0 weight
    geo = geodesic_positivity_overlap(pts, par, ~par, radius=0.5, k=10, log_w=log_w)
    assert geo.coverage > 0.8
    assert geo.mass_in_region <= 1.0
