# analysis/metrics.py
from __future__ import annotations
import numpy as np


def gini(x: np.ndarray) -> float:
    """
    x must be non-negative.
    """
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return 0.0
    if np.allclose(x.sum(), 0.0):
        return 0.0
    x = np.sort(x)
    n = x.size
    cumx = np.cumsum(x)
    # Gini = 1 - 2 * area under Lorenz curve
    lorenz_area = (cumx / cumx[-1]).sum() / n
    return 1.0 - 2.0 * (lorenz_area - (0.5 / n))


def top_share(x: np.ndarray, frac: float) -> float:
    """
    frac=0.01 => top 1% share
    """
    x = np.asarray(x, dtype=float)
    if x.size == 0 or np.allclose(x.sum(), 0.0):
        return 0.0
    n = x.size
    k = max(1, int(np.ceil(frac * n)))
    xs = np.sort(x)[::-1]
    return float(xs[:k].sum() / xs.sum())


def lorenz_curve(x: np.ndarray):
    x = np.asarray(x, dtype=float)
    x = np.clip(x, 0, None)
    x = np.sort(x)
    if x.sum() == 0:
        p = np.linspace(0, 1, x.size + 1)
        L = np.zeros_like(p)
        return p, L
    cum = np.cumsum(x)
    L = np.concatenate([[0.0], cum / cum[-1]])
    p = np.linspace(0, 1, x.size + 1)
    return p, L
