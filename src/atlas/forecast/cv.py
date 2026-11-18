"""Walk-forward training scheme with purging and embargo (Spec §7.2; Ref 3.9).

* Expanding window: train on all dates up to T, predict the next block, add it, repeat (monthly).
* Purging: a training row dated t has target window [t+1, t+h]. It is removed when that window
  overlaps the test block, i.e. when t + h >= first test date (in trading days).
* Embargo: a further h trading days after each test block are skipped before training resumes. In a
  purely expanding scheme the test block is always after the training data, so the embargo matters
  for the inner (nested) cross-validation used to tune hyperparameters, where folds can lie on both
  sides of the validation block.

All arithmetic is in trading-day positions on the sorted list of unique dates, so weekends and
holidays do not distort the gap.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd


@dataclass
class Split:
    train_dates: np.ndarray
    test_dates: np.ndarray


def walk_forward_splits(dates, h: int, min_train: int = 504, test_size: int = 21,
                        embargo: int | None = None) -> Iterator[Split]:
    """Expanding-window splits over trading days with purging of overlapping targets.

    dates: all trading days of the panel (any order, duplicates allowed). Training rows whose target
    window reaches the test block are purged: train dates must satisfy pos(t) + h < pos(first test).
    """
    u = np.array(sorted(pd.unique(pd.Series(dates))))
    start = min_train
    while start < len(u):
        test = u[start:start + test_size]
        train_end = start - h                 # purge: last allowed training position is start - h - 1
        if train_end > 0:
            yield Split(u[:train_end], test)
        start += test_size


def purged_kfold(dates, h: int, n_splits: int = 5, embargo: int | None = None) -> Iterator[Split]:
    """Purged k-fold with embargo for hyperparameter tuning inside a training window.

    Each fold is a contiguous block of dates. Training rows are dropped when their target window
    [t+1, t+h] overlaps the validation block (before it) or when they fall within `embargo` days after
    it (after it), so information cannot leak in either direction.
    """
    embargo = h if embargo is None else embargo
    u = np.array(sorted(pd.unique(pd.Series(dates))))
    folds = np.array_split(np.arange(len(u)), n_splits)
    for f in folds:
        lo, hi = f[0], f[-1]
        pos = np.arange(len(u))
        before = pos + h < lo                 # target window ends before validation starts
        after = pos > hi + embargo            # beyond the embargo period
        yield Split(u[before | after], u[f])


def leakage_check(train_dates, test_dates, all_dates, h: int) -> bool:
    """True when no training target window [t+1, t+h] reaches into the test block."""
    u = np.array(sorted(pd.unique(pd.Series(all_dates))))
    pos = {d: i for i, d in enumerate(u)}
    first_test = min(pos[d] for d in test_dates)
    last_test = max(pos[d] for d in test_dates)
    for d in train_dates:
        p = pos[d]
        if p < first_test and p + h >= first_test:
            return False
        if first_test <= p <= last_test:
            return False
    return True
