"""Shared cross-fitting fold assignment, with provenance-aware grouping.

Cross-fitting's variance theory assumes folds are drawn independently. If
some rows are not independent of others (e.g. a synthetic unit generated
conditioned on / near a specific real unit — see causal_bench.dgp.augmentation),
splitting them into different folds under the ordinary iid scheme silently
breaks that assumption: the "out-of-fold" prediction for one of them is not
really out-of-fold with respect to information about the other, the IC-based
variance is too small, and CI coverage drifts below nominal.

mode="iid" is today's existing behavior (StratifiedKFold for classification,
KFold for regression), unaware of any grouping. mode="group" uses sklearn's
GroupKFold so every row sharing a `groups` id is guaranteed to land in the
same fold, restoring the independence cross-fitting needs.
"""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold


def make_folds(
    X,
    y: np.ndarray | None = None,
    n_folds: int = 5,
    mode: str = "iid",
    groups: np.ndarray | None = None,
    random_state: int | None = None,
    stratify: bool = False,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return [(train_idx, val_idx), ...] for the requested fold mode.

    mode="iid": ignores `groups` entirely — current/default behavior.
        Uses StratifiedKFold(y) if stratify and y is not None, else KFold.
    mode="group": requires `groups`; uses GroupKFold so no group is split
        across folds. (GroupKFold has no stratified variant — `stratify` is
        ignored in this mode.)
    """
    if mode == "group":
        if groups is None:
            raise ValueError("mode='group' requires `groups` (e.g. provenance_group)")
        splitter = GroupKFold(n_splits=n_folds)
        return list(splitter.split(X, y, groups=groups))
    elif mode == "iid":
        if stratify and y is not None:
            splitter = StratifiedKFold(n_splits=n_folds, shuffle=True,
                                        random_state=random_state)
        else:
            splitter = KFold(n_splits=n_folds, shuffle=True,
                              random_state=random_state)
        return list(splitter.split(X, y))
    else:
        raise ValueError(f"Unknown fold mode: {mode!r}")
