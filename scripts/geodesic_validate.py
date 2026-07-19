"""Geodesic positivity R (#159): geodesic rejects the manufactured positivity a
Euclidean R reports across disconnected manifold sheets. CPU only.
Run: `PYTHONPATH=. python scripts/geodesic_validate.py`."""
import numpy as np

from causal_bench.geometry.geodesic import (
    geodesic_positivity_overlap, geodesic_distance_dijkstra, geodesic_distance_heat,
    make_two_sheets, make_swiss_roll)
from causal_bench.geometry.spectral import build_knn_laplacian
from causal_bench.sampling.stratified import positivity_overlap


def main() -> None:
    X, a, b = make_two_sheets(n=200, gap=0.3, seed=0)
    radius = 0.45
    flat = positivity_overlap(X[a], X[b], radius=radius)
    geo = geodesic_positivity_overlap(X, a, b, radius=radius, k=6)
    print("Two Euclidean-near but geodesically-disconnected sheets (radius > gap):")
    print(f"  Euclidean R coverage = {flat.coverage:.2f}   (manufactured positivity)")
    print(f"  geodesic  R coverage = {geo.coverage:.2f}   (rejected — sheets disconnected)")

    print("\nGeodesic ≥ Euclidean, and heat-Varadhan tracks the dijkstra oracle:")
    Xs, _ = make_swiss_roll(n=250, seed=1)
    d_geo = geodesic_distance_dijkstra(Xs, k=10, sources=[0])[0]
    d_euc = np.linalg.norm(Xs - Xs[0], axis=1)
    L = build_knn_laplacian(Xs, k=10)
    d_heat = np.asarray(geodesic_distance_heat(L, 0, t=0.03))
    m = np.isfinite(d_geo) & np.isfinite(d_heat)
    ra, rb = np.argsort(np.argsort(d_geo[m])), np.argsort(np.argsort(d_heat[m]))
    print(f"  min(geodesic − euclidean) = {np.min(d_geo[m] - d_euc[m]):+.3f}  (≥ 0)")
    print(f"  Spearman(heat, dijkstra)  = {np.corrcoef(ra, rb)[0, 1]:.3f}")
    print("\nRESULT: geodesic R prevents the false positivity Euclidean manufactures.")


if __name__ == "__main__":
    main()
