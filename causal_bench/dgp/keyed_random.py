import hashlib


def keyed_uniform(patient_id: int, event_type: str,
                  scenario: str, seed: int) -> float:
    """Deterministic Uniform(0,1) draw keyed by (patient_id, event_type, scenario, seed).

    Same key → same value across all calls. Different key components → independent values.
    Reference: Buffalo et al. (2026).
    """
    key = f"{seed}:{patient_id}:{event_type}:{scenario}"
    h = hashlib.sha256(key.encode()).digest()
    return int.from_bytes(h[:8], "big") / (2**64)


def keyed_normal(patient_id: int, event_type: str,
                 scenario: str, seed: int) -> float:
    """Standard Normal draw via inverse CDF of keyed_uniform."""
    from scipy.stats import norm
    u = keyed_uniform(patient_id, event_type, scenario, seed)
    u = max(1e-10, min(1 - 1e-10, u))
    return float(norm.ppf(u))
