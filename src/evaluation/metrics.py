"""The official F0.5 metric, implemented exactly as specified.

Definition
----------
For one Source 1 entity with ground-truth set ``G`` and predicted set ``P``:

* ``G == {}`` and ``P == {}``  -> ``1.0``   (correctly left unmatched)
* ``G == {}`` and ``P != {}``  -> ``0.0``   (false merge of a singleton)
* ``G != {}`` and ``P == {}``  -> ``0.0``   (missed entity)
* otherwise, with ``tp = |G & P|``::

      precision = tp / |P|
      recall    = tp / |G|
      F0.5      = 1.25 * precision * recall / (0.25 * precision + recall)

The final score is the **macro average over all Source 1 entities**, i.e.
entities with no true match count exactly as much as entities with many.

Two implementations are provided:

``f05_score``
    A direct, obviously-correct reference implementation used by the unit
    tests.  O(total matches).
``F05Evaluator``
    A sweep-based implementation used for threshold search.  It exploits the
    fact that the per-entity score depends only on the integer triple
    ``(tp, n_pred, n_gt)``, so all thresholds can be evaluated in a single
    O(n log n) pass instead of O(n_thresholds * n_candidates).
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import numpy as np

LOGGER = logging.getLogger(__name__)

BETA = 0.5
BETA_SQ = BETA * BETA  # 0.25


# --------------------------------------------------------------------------
# Reference implementation
# --------------------------------------------------------------------------
def f05_per_entity(gt_matches: Iterable[str], predicted: Iterable[str]) -> float:
    """F0.5 for a single Source 1 entity (reference implementation)."""
    gt = set(gt_matches)
    pred = set(predicted)
    if not gt:
        return 1.0 if not pred else 0.0
    if not pred:
        return 0.0
    tp = len(gt & pred)
    if tp == 0:
        return 0.0
    precision = tp / len(pred)
    recall = tp / len(gt)
    return (1.0 + BETA_SQ) * precision * recall / (BETA_SQ * precision + recall)


def f05_score(ground_truth: Mapping[str, Iterable[str]], predictions: Mapping[str, Iterable[str]]) -> float:
    """Macro-averaged F0.5 over the Source 1 entities in ``ground_truth``."""
    if not ground_truth:
        return 0.0
    total = 0.0
    for s1_id, gt in ground_truth.items():
        total += f05_per_entity(gt, predictions.get(s1_id, ()))
    return total / len(ground_truth)


# --------------------------------------------------------------------------
# Sweep implementation (used for threshold optimisation)
# --------------------------------------------------------------------------
def f05_from_counts(tp: int, n_pred: int, n_gt: int) -> float:
    """F0.5 from the integer triple; mirrors :func:`f05_per_entity` exactly."""
    if n_gt == 0:
        return 1.0 if n_pred == 0 else 0.0
    if n_pred == 0 or tp == 0:
        return 0.0
    precision = tp / n_pred
    recall = tp / n_gt
    return (1.0 + BETA_SQ) * precision * recall / (BETA_SQ * precision + recall)


@dataclass
class ThresholdPoint:
    """Metrics for one decision threshold."""

    threshold: float
    f05: float
    macro_precision: float
    macro_recall: float
    n_pred_pairs: int
    n_entities_with_prediction: int
    n_singleton_violations: int
    tp: int
    fp: int
    fn: int

    def as_row(self) -> dict[str, float]:
        return {
            "threshold": round(float(self.threshold), 6),
            "f05": round(float(self.f05), 6),
            "macro_precision": round(float(self.macro_precision), 6),
            "macro_recall": round(float(self.macro_recall), 6),
            "n_pred_pairs": int(self.n_pred_pairs),
            "tp": int(self.tp), "fp": int(self.fp), "fn": int(self.fn),
            "entities_with_prediction": int(self.n_entities_with_prediction),
            "singleton_violations": int(self.n_singleton_violations),
        }


@dataclass
class F05Evaluator:
    """Incremental F0.5 evaluator over scored candidate pairs.

    Parameters
    ----------
    entity_ids:
        All Source 1 entities that participate in the evaluation.  Every one of
        them contributes to the macro average, including entities with no
        candidates and entities whose ground truth is empty.
    truth:
        ``source1_id -> set of true match ids``.
    entity_pos:
        Optional map from ``source1_id`` to a contiguous integer index.  When
        omitted, indices are assigned in the order of ``entity_ids``.
    """

    entity_ids: Sequence[str]
    truth: Mapping[str, frozenset[str] | set[str]]
    entity_pos: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.entity_ids = [str(e) for e in self.entity_ids]
        if not self.entity_pos:
            self.entity_pos = {e: i for i, e in enumerate(self.entity_ids)}
        self.n_entities = len(self.entity_ids)
        # Map a match id to a truth bit per entity, stored as a dict keyed by
        # (entity_index, match_id) to keep memory proportional to true pairs.
        self._is_truth: set[tuple[int, str]] = set()
        self.n_gt: np.ndarray = np.zeros(self.n_entities, dtype=np.int64)
        for eid in self.entity_ids:
            idx = self.entity_pos[eid]
            matches = self.truth.get(eid) or ()
            self.n_gt[idx] = len(matches)
            for m in matches:
                self._is_truth.add((idx, str(m)))

    # -- per-threshold ----------------------------------------------------
    def label_pairs(
        self,
        entity_index: np.ndarray,
        match_id: Sequence[str],
        cache: dict[tuple[int, int, str], np.ndarray] | None = None,
    ) -> np.ndarray:
        """0/1 labels aligned with the pair arrays (vectorised, cacheable)."""
        key = (len(entity_index), int(entity_index[:5].sum()) if len(entity_index) else 0, str(match_id[0]) if len(match_id) else "")
        if cache is not None and key in cache:
            return cache[key]
        ent = np.asarray(entity_index, dtype=np.int64)
        mids = np.asarray(match_id, dtype=object)
        flags = np.zeros(len(ent), dtype=np.int64)
        for i in range(len(ent)):
            if (int(ent[i]), str(mids[i])) in self._is_truth:
                flags[i] = 1
        if cache is not None:
            cache[key] = flags
        return flags

    def predictions_from_scores(
        self, entity_index: np.ndarray, match_id: Sequence[str], scores: np.ndarray, threshold: float
    ) -> dict[str, list[str]]:
        """Materialise ``source1_id -> predicted ids`` for a fixed threshold."""
        keep = np.asarray(scores) >= threshold
        out: dict[str, list[str]] = {eid: [] for eid in self.entity_ids}
        for idx, mid in zip(np.asarray(entity_index)[keep].tolist(), np.asarray(match_id, dtype=object)[keep].tolist()):
            out[self.entity_ids[idx]].append(str(mid))
        return out

    def counts_from_threshold(
        self,
        entity_index: np.ndarray,
        labels: np.ndarray,
        scores: np.ndarray,
        threshold: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Per-entity ``(tp, n_pred)`` for a threshold, fully vectorised."""
        keep = np.asarray(scores) >= threshold
        ent = np.asarray(entity_index, dtype=np.int64)[keep]
        lab = np.asarray(labels, dtype=np.int64)[keep]
        n_pred = np.bincount(ent, minlength=self.n_entities).astype(np.int64)
        if lab.size:
            tp = np.bincount(ent, weights=lab, minlength=self.n_entities).astype(np.int64)
        else:
            tp = np.zeros(self.n_entities, dtype=np.int64)
        return tp, n_pred

    def evaluate_threshold(
        self, entity_index: np.ndarray, match_id: Sequence[str], scores: np.ndarray, threshold: float
    ) -> ThresholdPoint:
        """Metrics for one threshold, using the same macro definition."""
        labels = self.label_pairs(entity_index, match_id)
        return self._aggregate_at(threshold, entity_index, labels, scores)

    def _aggregate_at(
        self,
        threshold: float,
        entity_index: np.ndarray,
        labels: np.ndarray,
        scores: np.ndarray,
    ) -> ThresholdPoint:
        """``evaluate_threshold`` with pre-computed labels."""
        tp, n_pred = self.counts_from_threshold(entity_index, labels, scores, threshold)
        return self._aggregate(threshold, tp, n_pred)

    def _aggregate(self, threshold: float, tp: np.ndarray, n_pred: np.ndarray) -> ThresholdPoint:
        n_gt = self.n_gt
        # per-entity precision / recall with the singleton rules
        scores = np.where(
            n_gt == 0,
            np.where(n_pred == 0, 1.0, 0.0),
            np.where((n_pred == 0) | (tp == 0), 0.0, 1.0),
        )
        pos_mask = n_gt > 0
        # Closed form for entities that have both predictions and true matches:
        #   F = 1.25*p*r / (0.25*p + r)  with  p = tp/n_pred, r = tp/n_gt
        #     = 1.25*tp / (0.25*n_gt + n_pred)
        # The singleton rules (n_gt == 0 or n_pred == 0) are handled by `scores`.
        denominator = 0.25 * n_gt + n_pred
        f_values = np.where(denominator > 0, 1.25 * tp / np.where(denominator > 0, denominator, 1.0), scores)
        f05 = float(f_values.mean()) if self.n_entities else 0.0

        prec = np.zeros(self.n_entities, dtype=np.float64)
        rec = np.zeros(self.n_entities, dtype=np.float64)
        has_pred = n_pred > 0
        prec[has_pred] = tp[has_pred] / n_pred[has_pred]
        with np.errstate(divide="ignore", invalid="ignore"):
            rec[pos_mask] = tp[pos_mask] / np.where(n_gt[pos_mask] > 0, n_gt[pos_mask], 1)
        macro_precision = float(prec[has_pred].mean()) if has_pred.any() else 1.0
        macro_recall = float(rec[pos_mask].mean()) if pos_mask.any() else 1.0
        return ThresholdPoint(
            threshold=float(threshold),
            f05=f05,
            macro_precision=macro_precision,
            macro_recall=macro_recall,
            n_pred_pairs=int(n_pred.sum()),
            n_entities_with_prediction=int(has_pred.sum()),
            n_singleton_violations=int(((n_gt == 0) & (n_pred > 0)).sum()),
            tp=int(tp.sum()),
            fp=int((n_pred - tp).sum()),
            fn=int((n_gt - tp).sum()),
        )

    # -- sweep ------------------------------------------------------------
    def sweep(
        self,
        entity_index: np.ndarray,
        match_id: Sequence[str],
        scores: np.ndarray,
        thresholds: Sequence[float],
    ) -> list[ThresholdPoint]:
        """Evaluate every threshold, sorted by ascending threshold.

        The pair labels are computed once and shared across the whole grid;
        recomputing them per threshold turns a 100-point sweep into millions
        of redundant Python-level lookups.
        """
        labels = self.label_pairs(entity_index, match_id)
        unique = sorted({float(t) for t in thresholds})
        if not unique:
            return []
        return [
            self._aggregate_at(t, entity_index, labels, scores)
            for t in unique
        ]

    def sweep_all_breakpoints(
        self,
        entity_index: np.ndarray,
        match_id: Sequence[str],
        scores: np.ndarray,
        *,
        max_points: int = 400,
    ) -> list[ThresholdPoint]:
        """Evaluate at every distinct observed score value (exact breakpoints).

        Because predictions only change when the threshold crosses an observed
        score, this is the *complete* set of achievable operating points -- no
        threshold can be missed by a coarse grid.

        Ties matter: several pairs can share a score, and at threshold ``v``
        *all* of them are predicted.  The running counts are therefore advanced
        over an entire tie group before the operating point is recorded.
        """
        scores = np.asarray(scores, dtype=np.float64)
        if scores.size == 0:
            return [self.evaluate_threshold(entity_index, match_id, scores, 0.5)]
        order = np.argsort(-scores, kind="stable")
        ent = np.asarray(entity_index, dtype=np.int64)[order]
        labels = self.label_pairs(entity_index, match_id)[order]
        sorted_scores = scores[order]

        # Distinct score values, descending.  When there are more of them than
        # `max_points`, keep an evenly spaced subset so the whole score range
        # stays covered -- truncating at the top would only explore the
        # high-precision end and silently hide the real optimum.
        distinct = np.unique(sorted_scores)
        if max_points and len(distinct) > max_points:
            picks = np.unique(np.linspace(0, len(distinct) - 1, max_points).astype(int))
            wanted = {float(distinct[i]) for i in picks}
        else:
            wanted = {float(v) for v in distinct}

        tp = np.zeros(self.n_entities, dtype=np.int64)
        n_pred = np.zeros(self.n_entities, dtype=np.int64)
        points: list[ThresholdPoint] = []
        n = len(order)
        i = 0
        while i < n:
            value = float(sorted_scores[i])
            j = i
            while j < n and float(sorted_scores[j]) == value:
                idx = int(ent[j])
                n_pred[idx] += 1
                tp[idx] += int(labels[j])
                j += 1
            if value in wanted:
                points.append(self._aggregate(value, tp.copy(), n_pred.copy()))
            i = j
        points.sort(key=lambda p: p.threshold)
        return points


def best_threshold(points: Sequence[ThresholdPoint], tie_break_eps: float = 1e-9) -> ThresholdPoint:
    """Pick the threshold maximising F0.5, breaking ties toward precision.

    Because F0.5 weights precision 4x more strongly than recall, an equally good
    but more conservative threshold is always preferred; ties are therefore
    resolved by the higher macro precision, then by the higher threshold.
    """
    if not points:
        raise ValueError("no threshold points to select from")
    best_f = max(p.f05 for p in points)
    contenders = [p for p in points if p.f05 >= best_f - tie_break_eps]
    return max(contenders, key=lambda p: (p.macro_precision, p.threshold))


def summarize_confusion(points: Sequence[ThresholdPoint]) -> str:
    """Human-readable threshold table."""
    header = f"{'thr':>6} {'F0.5':>8} {'P':>7} {'R':>7} {'pairs':>10} {'TP':>8} {'FP':>8} {'FN':>8} {'singletonFP':>12}"
    lines = [header, "-" * len(header)]
    for p in points:
        lines.append(
            f"{p.threshold:6.3f} {p.f05:8.4f} {p.macro_precision:7.4f} {p.macro_recall:7.4f} "
            f"{p.n_pred_pairs:10d} {p.tp:8d} {p.fp:8d} {p.fn:8d} {p.n_singleton_violations:12d}"
        )
    return "\n".join(lines)
