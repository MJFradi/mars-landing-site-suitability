"""Analytic Hierarchy Process: weights and consistency ratio (Saaty 1980)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Saaty's random consistency index for n = 1..10
RANDOM_INDEX = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24,
                7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}


@dataclass
class AHPResult:
    criteria: list[str]
    matrix: np.ndarray
    weights: dict[str, float]
    lambda_max: float
    ci: float
    cr: float

    @property
    def consistent(self) -> bool:
        return self.cr < 0.10

    def table(self) -> pd.DataFrame:
        df = pd.DataFrame(self.matrix, index=self.criteria, columns=self.criteria)
        df["weight"] = [self.weights[c] for c in self.criteria]
        return df

    def __str__(self) -> str:
        w = ", ".join(f"{c} {v:.3f}" for c, v in self.weights.items())
        return (f"weights: {w}\nlambda_max = {self.lambda_max:.3f}, CI = {self.ci:.4f}, "
                f"CR = {self.cr:.4f} ({'OK' if self.consistent else 'INCONSISTENT, revise'})")


def matrix_from_upper(criteria: list[str], upper: dict[tuple[str, str], float]) -> np.ndarray:
    """Build the full reciprocal matrix from upper-triangle judgements."""
    n = len(criteria)
    idx = {c: i for i, c in enumerate(criteria)}
    A = np.ones((n, n))
    for (a, b), v in upper.items():
        i, j = idx[a], idx[b]
        A[i, j], A[j, i] = v, 1.0 / v
    missing = [(criteria[i], criteria[j]) for i in range(n) for j in range(i + 1, n)
               if (criteria[i], criteria[j]) not in upper and (criteria[j], criteria[i]) not in upper]
    if missing:
        raise ValueError(f"missing judgements: {missing}")
    return A


def ahp(A: np.ndarray, criteria: list[str]) -> AHPResult:
    """Principal-eigenvector weights and consistency ratio for matrix ``A``."""
    A = np.asarray(A, dtype="float64")
    n = A.shape[0]
    if not np.allclose(A * A.T, 1.0):
        raise ValueError("matrix is not reciprocal (a_ij * a_ji must be 1)")
    vals, vecs = np.linalg.eig(A)
    k = int(np.argmax(vals.real))
    w = np.abs(vecs[:, k].real)
    w /= w.sum()
    lam = float(vals.real[k])
    ci = (lam - n) / (n - 1) if n > 1 else 0.0
    ri = RANDOM_INDEX.get(n)
    cr = ci / ri if ri else 0.0
    return AHPResult(list(criteria), A, dict(zip(criteria, map(float, w))), lam, ci, cr)
