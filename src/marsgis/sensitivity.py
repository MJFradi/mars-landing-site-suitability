"""Weight sensitivity: one-at-a-time, named scenarios and Dirichlet Monte Carlo."""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from . import config as C
from .grid import Grid
from .overlay import candidate_zones, weighted_overlay


def perturb(weights: dict[str, float], name: str, change: float) -> dict[str, float]:
    """Scale ``weights[name]`` by (1 + change) and rescale the others so the total stays 1.

    The other weights keep their ratios to each other.
    """
    w = dict(weights)
    new = min(max(w[name] * (1 + change), 0.0), 1.0)
    rest = sum(v for k, v in w.items() if k != name)
    for k in w:
        w[k] = new if k == name else w[k] * (1 - new) / rest
    return w


def scenario(weights: dict[str, float], fixed: dict[str, float]) -> dict[str, float]:
    """Pin some criteria to ``fixed`` shares; spread the remainder over the rest
    in proportion to ``weights``. Criteria in ``fixed`` but not in ``weights`` are added."""
    free = {k: v for k, v in weights.items() if k not in fixed}
    left = 1.0 - sum(fixed.values())
    tot = sum(free.values())
    out = {k: v * left / tot for k, v in free.items()}
    out.update(fixed)
    return out


def named_scenarios(weights: dict[str, float], science: bool = False) -> dict[str, dict[str, float]]:
    """The guide's three scenarios. "Science first" needs a ``science`` score layer."""
    s = {
        "AHP (baseline)": dict(weights),
        "Safety first": scenario(weights, {"slope": 0.35, "rough": 0.35}),
        "Equal weights": {k: 1 / len(weights) for k in weights},
    }
    if science:
        s["Science first"] = scenario(weights, {"science": 0.40})
    return s


def top_overlap(base_ranked: np.ndarray, new_ranked: np.ndarray) -> int:
    """How many of the baseline top-N zones overlap any top-N zone of the new run."""
    hits = np.unique(base_ranked[(base_ranked > 0) & (new_ranked > 0)])
    return int(hits.size)


def run(scores, weights, mask, grid: Grid, threshold: float,
        base_ranked: np.ndarray | None = None,
        validate_fn: Callable[[np.ndarray], float] | None = None) -> dict:
    """One overlay run with its zones and metrics.

    ``threshold`` should stay fixed across runs (e.g. the baseline's "very
    high" class break) or be recomputed per run; see ``threshold_fn`` in
    ``one_at_a_time``. ``validate_fn(S)`` returns the mean site percentile.
    """
    S = weighted_overlay(scores, weights, mask)
    table, ranked = candidate_zones(S, grid, threshold)
    out = {"S": S, "table": table, "ranked": ranked}
    if base_ranked is not None:
        out["top_overlap"] = top_overlap(base_ranked, ranked)
    if validate_fn is not None:
        out["mean_site_percentile"] = validate_fn(S)
    return out


def quantile_threshold(S: np.ndarray, q: float = 0.8) -> float:
    """Lower bound of the top (1 - q) of unmasked cells, i.e. the "very high" class."""
    v = S[np.isfinite(S) & (S > 0)]
    return float(np.quantile(v, q))


def one_at_a_time(scores, weights, mask, grid: Grid, base_ranked,
                  changes=(-0.4, -0.2, 0.2, 0.4), validate_fn=None,
                  threshold_fn=quantile_threshold) -> pd.DataFrame:
    rows = []
    for name in weights:
        for ch in changes:
            w = perturb(weights, name, ch)
            S = weighted_overlay(scores, w, mask)
            r = run(scores, w, mask, grid, threshold_fn(S), base_ranked, validate_fn)
            rows.append({"criterion": name, "change": ch, "weight": w[name],
                         "top_overlap": r["top_overlap"],
                         "mean_site_percentile": r.get("mean_site_percentile", np.nan)})
    for name in weights:                                   # the 0 % point on every line
        rows.append({"criterion": name, "change": 0.0, "weight": weights[name],
                     "top_overlap": int(base_ranked.max()), "mean_site_percentile": np.nan})
    return pd.DataFrame(rows).sort_values(["criterion", "change"]).reset_index(drop=True)


def monte_carlo(scores, weights, mask, n_runs: int = 500, concentration: float = 50.0,
                top_frac: float = 0.10, seed: int = C.SEED) -> tuple[np.ndarray, np.ndarray]:
    """Probability that each cell is in the top ``top_frac`` of unmasked cells.

    Weights are drawn from a Dirichlet centred on the AHP weights
    (alpha = concentration * w); higher concentration = smaller spread.
    Works on the 1-D vector of unmasked cells to keep memory down.
    Returns (probability raster, drawn weights of shape (n_runs, n_criteria)).
    """
    rng = np.random.default_rng(seed)
    names = list(weights)
    alpha = concentration * np.array([weights[k] for k in names])
    draws = rng.dirichlet(alpha, size=n_runs)

    cell = mask & np.all([np.isfinite(scores[k]) for k in names], axis=0)
    stack = np.stack([scores[k][cell] for k in names]).astype("float32")   # (k, N)
    k_top = max(1, int(round(top_frac * stack.shape[1])))
    hits = np.zeros(stack.shape[1], dtype="uint16")
    for w in draws:
        s = w.astype("float32") @ stack
        cut = np.partition(s, -k_top)[-k_top]
        hits += s >= cut
    has_data = np.all([np.isfinite(scores[k]) for k in names], axis=0)
    prob = np.where(has_data, 0.0, np.nan).astype("float32")   # excluded ground = 0
    prob[cell] = hits / n_runs
    return prob, draws
