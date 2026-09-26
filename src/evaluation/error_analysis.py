"""Error analysis: why the pipeline loses F0.5.

Every error is attributed to exactly one of four causes, and the split matters
because the fixes are completely different:

``blocking_failure``
    The true match was never generated as a candidate.  No model can recover
    it -- the fix is in retrieval.
``model_failure``
    The true match was a candidate but scored below the threshold.  The fix is
    in features / model / threshold.
``false_positive``
    A non-matching candidate scored above the threshold.  Under F0.5 this is
    the most expensive error, because for a true singleton it scores 0.
``singleton_violation``
    A Source 1 entity whose ground truth is empty received at least one match.
    This is a special case of ``false_positive`` with the highest cost, so it
    is reported separately.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from ..blocking.candidate_generator import CandidateSet, strategies_from_mask
from ..config import ID_COLUMN
from ..data.ground_truth import GroundTruth
from ..models.predict import MatchingResult

LOGGER = logging.getLogger(__name__)


@dataclass
class ErrorAnalysis:
    """Bucketed error report with representative examples."""

    threshold: float
    n_pairs: int
    counts: dict[str, int] = field(default_factory=dict)
    rates: dict[str, float] = field(default_factory=dict)
    score_distribution: dict[str, dict[str, float]] = field(default_factory=dict)
    examples: dict[str, list[dict]] = field(default_factory=dict)
    worst_entities: list[dict] = field(default_factory=list)

    def render(self) -> str:
        lines = ["ERROR ANALYSIS", "-" * 92, f"  decision threshold: {self.threshold:.4f}"]
        lines.append(f"  {'bucket':<24} {'count':>10} {'rate':>10}")
        for key, value in self.counts.items():
            lines.append(f"  {key:<24} {value:>10} {self.rates.get(key, 0.0):>9.2%}")
        if self.score_distribution:
            lines.append("\n  score distribution by truth (mean / p10 / p50 / p90 / p99 / max):")
            for key, stats in self.score_distribution.items():
                lines.append(
                    f"    {key:<28} {stats['mean']:.4f}  {stats['p10']:.4f}  {stats['p50']:.4f}  "
                    f"{stats['p90']:.4f}  {stats['p99']:.4f}  {stats['max']:.4f}"
                )
        for bucket, rows in self.examples.items():
            if not rows:
                continue
            lines.append(f"\n  --- {bucket} (showing {len(rows)}) ---")
            for row in rows:
                lines.append("    " + json.dumps(row, default=str))
        if self.worst_entities:
            lines.append("\n  --- entities contributing the most lost F0.5 ---")
            for row in self.worst_entities:
                lines.append("    " + json.dumps(row, default=str))
        return "\n".join(lines)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.__dict__, indent=2, default=str), encoding="utf-8")


def _record(frame_lookup: dict[str, pd.Series], entity_id: str, limit: int = 90) -> dict:
    row = frame_lookup.get(entity_id)
    if row is None:
        return {"entity_id": entity_id}
    out = {"entity_id": entity_id}
    for key, value in row.items():
        text = "" if value is None else str(value)
        out[key] = text[:limit]
    return out


def analyse_errors(
    candidates: CandidateSet,
    result: MatchingResult,
    ground_truth: GroundTruth,
    scores: np.ndarray,
    labels: np.ndarray,
    source1_ids: Sequence[str],
    record_lookup: dict[str, pd.Series],
    candidate_lookup: dict[str, pd.Series],
    feature_frame: pd.DataFrame | None = None,
    feature_columns: Sequence[str] = (),
    *,
    threshold: float | None = None,
    max_examples: int = 8,
) -> ErrorAnalysis:
    """Attribute every remaining error to a blocking or a model failure."""
    threshold = result.threshold if threshold is None else threshold
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    s1_arr = np.asarray([str(s) for s in candidates.query_ids[candidates.query_pos]], dtype=object)
    c_arr = np.asarray([str(c) for c in candidates.pool_ids[candidates.pool_pos]], dtype=object)

    predicted_mask = scores >= threshold
    is_true = labels == 1

    # --- which true matches never became candidates -----------------------
    generated: dict[str, set[str]] = {}
    for s1, cid in zip(s1_arr.tolist(), c_arr.tolist()):
        generated.setdefault(s1, set()).add(cid)
    blocking_failures: list[dict] = []
    n_blocking_failed_pairs = 0
    for s1 in source1_ids:
        truth = ground_truth.get(str(s1))
        if not truth:
            continue
        missing = truth - generated.get(str(s1), set())
        if missing:
            n_blocking_failed_pairs += len(missing)
            blocking_failures.append(
                {
                    "source1": _record(record_lookup, str(s1)),
                    "n_true": len(truth),
                    "missing_from_candidates": sorted(missing)[:5],
                    "candidate_examples": [
                        _record(candidate_lookup, m) for m in sorted(generated.get(str(s1), ()))[:2]
                    ],
                }
            )

    # --- model failures / false positives ---------------------------------
    model_failures: list[dict] = []
    false_positives: list[dict] = []
    singleton_violations: list[dict] = []
    n_model_failures = int((is_true & ~predicted_mask).sum())
    n_false_positives = int((~is_true & predicted_mask).sum())
    n_singleton = 0
    gt_sizes = {s: len(ground_truth.get(s)) for s in source1_ids}
    for idx in np.flatnonzero(is_true & ~predicted_mask)[: max_examples * 3]:
        model_failures.append(
            {
                "source1": _record(record_lookup, str(s1_arr[idx])),
                "candidate": _record(candidate_lookup, str(c_arr[idx])),
                "score": round(float(scores[idx]), 6),
                "gap_to_threshold": round(threshold - float(scores[idx]), 6),
                "top_features": _top_features(feature_frame, idx, feature_columns),
            }
        )
    for idx in np.flatnonzero(~is_true & predicted_mask)[: max_examples * 3]:
        s1 = str(s1_arr[idx])
        row = {
            "source1": _record(record_lookup, s1),
            "candidate": _record(candidate_lookup, str(c_arr[idx])),
            "score": round(float(scores[idx]), 6),
            "n_true_matches_for_source1": gt_sizes.get(s1, 0),
            "candidate_strategies": strategies_from_mask(int(candidates.strategy_mask[idx])),
            "top_features": _top_features(feature_frame, idx, feature_columns),
        }
        if gt_sizes.get(s1, 0) == 0:
            row["type"] = "singleton_violation"
            singleton_violations.append(row)
        false_positives.append(row)

    # --- lost F0.5 per entity --------------------------------------------
    entity_loss: list[dict] = []
    for s1 in source1_ids:
        truth = ground_truth.get(str(s1))
        pred = set(result.predictions.get(str(s1), ()))
        gain = _f05(truth, pred)
        if gain < 1.0:
            false_merges = sorted(pred - truth)
            entity_loss.append(
                {
                    "source1_entity_id": str(s1),
                    "n_true": len(truth),
                    "n_predicted": len(pred),
                    "entity_f05": round(gain, 6),
                    "n_false_merges": len(false_merges),
                    "false_merges": false_merges[:4],
                    "missed_matches": sorted(truth - pred)[:4],
                }
            )
    # Worst first: lowest entity F0.5, then most false merges (costliest).
    entity_loss.sort(key=lambda r: (r["entity_f05"], -r["n_false_merges"]))

    n_singleton_violations = sum(
        1 for idx in np.flatnonzero(~is_true & predicted_mask) if gt_sizes.get(str(s1_arr[idx]), 0) == 0
    )
    counts = {
        "true pairs not generated": n_blocking_failed_pairs,
        "true pairs scored too low": n_model_failures,
        "false positives (predicted, not true)": n_false_positives,
        "of which singleton violations": n_singleton_violations,
        "entities with imperfect F0.5": len(entity_loss),
        "total candidate pairs": int(len(scores)),
    }
    n_true_total = max(1, int(is_true.sum()))
    rates = {
        "true pairs not generated": n_blocking_failed_pairs / n_true_total,
        "true pairs scored too low": n_model_failures / n_true_total,
        "false positives (predicted, not true)": n_false_positives / max(1, int(predicted_mask.sum())),
    }
    distribution = {}
    for key, mask in (("true pairs", is_true), ("false pairs", ~is_true)):
        if mask.any():
            values = scores[mask]
            distribution[key] = {
                "mean": float(values.mean()), "p10": float(np.percentile(values, 10)),
                "p50": float(np.percentile(values, 50)), "p90": float(np.percentile(values, 90)),
                "p99": float(np.percentile(values, 99)), "max": float(values.max()),
            }
    return ErrorAnalysis(
        threshold=threshold,
        n_pairs=int(len(scores)),
        counts=counts,
        rates=rates,
        score_distribution=distribution,
        examples={
            "blocking failures": blocking_failures[:max_examples],
            "model failures (true match ranked too low)": model_failures[:max_examples],
            "false positives": false_positives[:max_examples],
            "singleton violations (entity had no true match)": singleton_violations[:max_examples],
        },
        worst_entities=entity_loss[:max_examples * 2],
    )


def _f05(truth: frozenset[str], predicted: set[str]) -> float:
    from .metrics import f05_per_entity

    return f05_per_entity(truth, predicted)


def _top_features(feature_frame: pd.DataFrame | None, idx: int, columns: Sequence[str], k: int = 6) -> dict:
    """The most active features for one pair -- a compact "why" explanation."""
    if feature_frame is None or not columns:
        return {}
    row = feature_frame.iloc[idx]
    values = {c: float(row[c]) for c in columns if c in row.index}
    top = sorted(values.items(), key=lambda kv: -abs(kv[1]))[:k]
    return {k: round(v, 4) for k, v in top if v != 0.0}


def summarise_entity_scores(
    scores: np.ndarray, labels: np.ndarray, buckets: Sequence[float] = (0.1, 0.3, 0.5, 0.7, 0.9)
) -> pd.DataFrame:
    """Precision of the model's own score, bucketed -- a calibration view."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    edges = list(buckets)
    rows = []
    for low, high in zip([0.0] + edges, edges + [1.01]):
        mask = (scores >= low) & (scores < high)
        if not mask.any():
            continue
        rows.append(
            {
                "score_bucket": f"[{low:.2f}, {high:.2f})",
                "n_pairs": int(mask.sum()),
                "n_positive": int(labels[mask].sum()),
                "empirical_precision": round(float(labels[mask].mean()), 6),
            }
        )
    return pd.DataFrame(rows)


def count_by_strategy(candidates: CandidateSet, labels: np.ndarray) -> Counter:
    """How many positives each strategy contributed (diagnostic only)."""
    out: Counter = Counter()
    for label, mask in zip(candidates.strategy_mask, labels == 1):
        if label:
            for name in strategies_from_mask(int(mask)):
                out[name] += 1
    return out
