"""Validation of the *whole* pipeline, and candidate-recall measurement.

Leakage control
---------------
The split is performed on **Source 1 entities**, which is the unit the metric
is macro-averaged over and the unit that must be unseen at test time::

    Source 1 entities
        |-- fold k       -> validation queries (never used to fit the model)
        `-- folds != k   -> training queries (produce the model's training pairs)

What is deliberately shared between the folds:

* the **retrieval pool** (Source 2 + Source 3), and
* the **TF-IDF / IDF statistics** fitted on that pool.

Both are label-free functions of the record data, and both are exactly what is
available at inference time (at test time the pool is the *test* pool).  Making
the pool fold-specific would instead *understate* both candidate recall and
F0.5, so it is deliberately not done.  No ground-truth label, and no statistic
computed from one, ever crosses a fold boundary.

Because the fold split is on Source 1 entities, and candidates are generated
once for all of them and then partitioned by query id, the feature matrix is
also identical to what inference would produce for unseen queries.
"""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from ..blocking.candidate_generator import STRATEGY_NAMES, CandidateSet, strategies_from_mask
from ..config import Config
from ..data.ground_truth import GroundTruth
from ..evaluation.metrics import F05Evaluator, ThresholdPoint, f05_score
from ..features.feature_builder import FeatureMatrix
from ..models.predict import decide_matches
from ..models.threshold import optimize_threshold
from ..models.train import MODEL_NAMES, TrainedModel, build_label_matrix, train_model

LOGGER = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Fold construction
# --------------------------------------------------------------------------
def make_entity_folds(
    source1_ids: Sequence[str], n_splits: int, random_state: int
) -> list[tuple[list[str], list[str]]]:
    """K-fold split over Source 1 entity ids (shuffled, seeded)."""
    ids = np.array([str(i) for i in source1_ids])
    n_splits = max(2, min(int(n_splits), len(ids)))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return [(sorted(ids[tr].tolist()), sorted(ids[va].tolist())) for tr, va in kf.split(ids)]


# --------------------------------------------------------------------------
# Candidate recall
# --------------------------------------------------------------------------
@dataclass
class CandidateRecall:
    """How much of the ground truth the blocking stage actually surfaces."""

    n_true_pairs: int
    n_found_pairs: int
    recall: float
    entities_with_truth: int
    entities_fully_recalled: int
    entities_partially_recalled: int
    per_strategy: dict[str, dict[str, float]] = field(default_factory=dict)
    per_source: dict[str, float] = field(default_factory=dict)
    per_cardinality: dict[str, dict[str, float]] = field(default_factory=dict)
    examples: list[dict] = field(default_factory=list)

    def as_row(self) -> dict:
        return {
            "n_true_pairs": self.n_true_pairs,
            "n_found_pairs": self.n_found_pairs,
            "candidate_recall": round(self.recall, 6),
            "entities_with_truth": self.entities_with_truth,
            "entities_fully_recalled": self.entities_fully_recalled,
            "entities_partially_recalled": self.entities_partially_recalled,
        }

    def render(self) -> str:
        lines = [
            "CANDIDATE RECALL (upper bound on achievable F0.5)",
            f"  true pairs                     : {self.n_true_pairs}",
            f"  true pairs present in candidates: {self.n_found_pairs}",
            f"  CANDIDATE RECALL               : {self.recall:.4f}",
            f"  entities with >=1 true match    : {self.entities_with_truth}",
            f"  fully recalled                  : {self.entities_fully_recalled}",
            f"  partially recalled              : {self.entities_partially_recalled}",
        ]
        if self.per_strategy:
            lines.append("  per-strategy isolated recall:")
            for name, stats in sorted(self.per_strategy.items(), key=lambda kv: -kv[1]["recall"]):
                lines.append(f"    {name:<18} {stats['found']:>8} / {self.n_true_pairs:<8} recall={stats['recall']:.4f}")
        if self.per_source:
            lines.append("  recall by candidate source:")
            for name, recall in sorted(self.per_source.items()):
                lines.append(f"    {name:<18} recall={recall:.4f}")
        if self.per_cardinality:
            lines.append("  recall by number of true matches:")
            for key, stats in sorted(self.per_cardinality.items(), key=lambda kv: int(kv[0])):
                lines.append(
                    f"    {key:>3} true match(es): {stats['found']:>7} / {stats['total']:<7} "
                    f"pair recall={stats['recall']:.4f}   "
                    f"fully-recalled entities {stats['entities_fully_recalled']}/{stats['entities']}"
                )
        return "\n".join(lines)


def measure_candidate_recall(
    candidates: CandidateSet,
    ground_truth: GroundTruth,
    source1_ids: Sequence[str],
    *,
    max_examples: int = 15,
) -> CandidateRecall:
    """Compare ground-truth matches against the generated candidate set."""
    generated: dict[str, set[str]] = defaultdict(set)
    masks: dict[str, dict[str, int]] = defaultdict(dict)
    for q, p, m in zip(candidates.query_pos, candidates.pool_pos, candidates.strategy_mask):
        s1 = str(candidates.query_ids[q])
        cid = str(candidates.pool_ids[p])
        generated[s1].add(cid)
        masks[s1][cid] = int(m)

    n_true = 0
    n_found = 0
    with_truth = fully = partial = 0
    strategy_found: Counter[str] = Counter()
    src_found: Counter[str] = Counter()
    src_total: Counter[str] = Counter()
    card_found: Counter[int] = Counter()
    card_total: Counter[int] = Counter()
    card_entities: Counter[int] = Counter()
    card_entities_full: Counter[int] = Counter()
    examples: list[dict] = []

    for s1 in source1_ids:
        truth = ground_truth.get(str(s1))
        n_true += len(truth)
        if not truth:
            continue
        with_truth += 1
        # Both the pair counts and the entity counts are tracked, so the
        # per-cardinality recall is a ratio of like with like.
        card_total[len(truth)] += len(truth)
        card_entities[len(truth)] += 1
        found_ids = truth & generated.get(str(s1), set())
        n_found += len(found_ids)
        card_found[len(truth)] += len(found_ids)
        if len(found_ids) == len(truth):
            fully += 1
            card_entities_full[len(truth)] += 1
        else:
            partial += 1
            if len(examples) < max_examples:
                examples.append(
                    {
                        "source1_entity_id": str(s1),
                        "n_true_matches": len(truth),
                        "n_found": len(found_ids),
                        "missing": sorted(truth - generated.get(str(s1), set())),
                        "n_candidates": len(generated.get(str(s1), ())),
                    }
                )
        for cid in found_ids:
            mask = masks.get(str(s1), {}).get(cid, 0)
            for name in strategies_from_mask(mask):
                strategy_found[name] += 1

    # Recall split by which source the true match came from.  The source is
    # read from the candidate pool itself -- never from an id prefix, because
    # entity id formats are not guaranteed across datasets.  Only the entities
    # in scope are counted, otherwise entities outside this run would sit in
    # the denominator and deflate the reported recall.
    pool_source = {str(pid): str(src) for pid, src in zip(candidates.pool_ids, candidates.pool_source)}
    in_scope = set(source1_ids)
    for s1, cid in ground_truth.positive_pairs():
        if str(s1) not in in_scope:
            continue
        src = pool_source.get(str(cid), "not_in_pool")
        src_total[src] += 1
        if cid in generated.get(str(s1), set()):
            src_found[src] += 1

    per_strategy = {
        name: {
            "found": int(strategy_found.get(name, 0)),
            "recall": round(strategy_found.get(name, 0) / n_true, 6) if n_true else 0.0,
        }
        for name in STRATEGY_NAMES
    }
    per_source = {
        k: (src_found.get(k, 0) / v if v else 0.0) for k, v in sorted(src_total.items())
    }
    per_cardinality = {
        str(k): {
            "found": int(card_found.get(k, 0)),
            "total": int(card_total.get(k, 0)),
            "recall": round(card_found.get(k, 0) / card_total[k], 6) if card_total.get(k, 0) else 0.0,
            "entities_fully_recalled": int(card_entities_full.get(k, 0)),
            "entities": int(card_entities.get(k, 0)),
        }
        for k in sorted(card_entities)
    }
    return CandidateRecall(
        n_true_pairs=n_true,
        n_found_pairs=n_found,
        recall=(n_found / n_true) if n_true else 0.0,
        entities_with_truth=with_truth,
        entities_fully_recalled=fully,
        entities_partially_recalled=partial,
        per_strategy=per_strategy,
        per_source=per_source,
        per_cardinality=per_cardinality,
        examples=examples,
    )


# --------------------------------------------------------------------------
# Cross-validated model comparison
# --------------------------------------------------------------------------
@dataclass
class ModelScore:
    """Out-of-fold performance of one candidate model."""

    name: str
    point: ThresholdPoint
    f05_by_fold: list[float]
    f05_mean: float
    f05_std: float
    threshold: float

    def as_row(self) -> dict:
        return {
            "model": self.name,
            "f05": round(self.point.f05, 6),
            "f05_mean_over_folds": round(self.f05_mean, 6),
            "f05_std_over_folds": round(self.f05_std, 6),
            "threshold": round(self.threshold, 6),
            "macro_precision": round(self.point.macro_precision, 6),
            "macro_recall": round(self.point.macro_recall, 6),
            "TP": self.point.tp, "FP": self.point.fp, "FN": self.point.fn,
        }


@dataclass
class ValidationReport:
    """Everything measured during validation, in one serialisable object."""

    n_source1: int
    n_pool: int
    folds: list[dict]
    candidate_recall: dict
    model_scores: list[dict]
    best_model: str
    best_threshold: float
    best_f05: float
    best_point: dict
    per_fold_f05: list[float]
    feature_importance: list[dict]
    blocking_stats: dict
    runtime_seconds: dict

    def render(self) -> str:
        n_folds = len({f.get("fold") for f in self.folds}) or len(self.folds)
        n_models = len(self.model_scores)
        lines = ["VALIDATION SUMMARY", "-" * 70,
                 f"  Source 1 entities            : {self.n_source1}",
                 f"  pool records (source2+3)     : {self.n_pool}",
                 f"  folds x models compared      : {n_folds} x {n_models}",
                 f"  candidate recall             : {self.candidate_recall.get('candidate_recall')}",
                 f"  selected model               : {self.best_model}",
                 f"  selected threshold           : {self.best_threshold:.4f}",
                 f"  out-of-fold F0.5             : {self.best_f05:.4f}",
                 f"  per-fold F0.5                : {[round(f, 4) for f in self.per_fold_f05]}",
                 "", "  model comparison (out-of-fold, threshold-optimised F0.5):"]
        for row in self.model_scores:
            lines.append(
                f"    {row['model']:<24} F0.5={row['f05']:.4f}  thr={row['threshold']:.3f}  "
                f"P={row['macro_precision']:.4f} R={row['macro_recall']:.4f}"
            )
        return "\n".join(lines)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.__dict__, indent=2, default=str), encoding="utf-8")


def _entity_index(ids: Sequence[str], universe: Sequence[str]) -> np.ndarray:
    lookup = {e: i for i, e in enumerate(universe)}
    return np.fromiter((lookup.get(str(e), 0) for e in ids), dtype=np.int64, count=len(ids))


def cross_validate(
    features: FeatureMatrix,
    candidates: CandidateSet,
    ground_truth: GroundTruth,
    config: Config,
) -> ValidationReport:
    """Run the full pipeline under K-fold cross-validation and rank the models."""
    import time

    t_start = time.perf_counter()
    source1_ids = [str(s) for s in features.source1_ids]
    # The scored universe is every Source 1 entity we were asked about --
    # including entities that produced no candidate at all.  Deriving it from
    # the feature rows would silently drop those entities and inflate F0.5.
    universe = sorted({str(s) for s in candidates.query_ids} | set(source1_ids))
    y = build_label_matrix(source1_ids, features.candidate_ids, ground_truth.matches)
    query_index = _entity_index(source1_ids, universe)

    LOGGER.info("Validation: %d candidate pairs, %d positives (%.3f%%), %d Source 1 entities",
                len(y), int(y.sum()), 100 * y.sum() / max(1, len(y)), len(universe))

    folds = make_entity_folds(universe, config.model.n_splits, config.random_state)
    fold_masks: list[np.ndarray] = []
    for train_ids, val_ids in folds:
        val_set = set(val_ids)
        is_val = np.fromiter((str(s) in val_set for s in source1_ids), dtype=bool, count=len(source1_ids))
        fold_masks.append(~is_val)

    recall = measure_candidate_recall(candidates, ground_truth, universe)
    print("\n" + "=" * 92)
    print(recall.render())
    print("=" * 92)

    model_names = [m for m in config.model.candidates if m in MODEL_NAMES]
    unknown = [m for m in config.model.candidates if m not in MODEL_NAMES]
    if unknown:
        raise ValueError(
            f"Unknown model name(s) {unknown}. Valid names are: {sorted(MODEL_NAMES)}"
        )
    if not model_names:
        raise ValueError(
            f"config.model.candidates is empty; choose from {sorted(MODEL_NAMES)}"
        )
    results: dict[str, ModelScore] = {}
    oof_store: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    fold_reports: list[dict] = []
    best_model_importance: dict[str, float] = {}

    for name in model_names:
        LOGGER.info("Cross-validating model '%s'", name)
        oof_scores = np.full(len(y), np.nan, dtype=np.float64)
        trained: list[TrainedModel] = []
        all_ids = np.asarray([str(s) for s in features.source1_ids])
        all_candidates = [str(c) for c in features.candidate_ids]
        for fold_index, (train_ids, val_ids) in enumerate(folds):
            train_mask = fold_masks[fold_index]
            if not train_mask.any() or not (~train_mask).any():
                continue
            model = train_model(name, features.X[train_mask], y[train_mask], features.columns, config.model)
            trained.append(model)
            oof_scores[~train_mask] = model.predict_proba(features.X[~train_mask])

        valid = ~np.isnan(oof_scores)
        if not valid.any():
            LOGGER.warning("Model '%s' produced no out-of-fold predictions", name)
            continue
        evaluator = F05Evaluator(universe, {e: ground_truth.get(e) for e in universe})
        threshold_result = optimize_threshold(
            query_index, all_candidates, oof_scores, evaluator, config.threshold, verbose=(name == model_names[0]),
        )
        best = threshold_result.best

        # Per-fold scores are measured at the *pooled* optimal threshold, so
        # the folds are directly comparable and consistent with the headline
        # number.  Optimising each fold separately would overstate stability.
        per_fold_f05: list[float] = []
        for fold_index, (train_ids, val_ids) in enumerate(folds):
            val_mask = ~fold_masks[fold_index]
            if not val_mask.any() or not valid[val_mask].any():
                continue
            fold_eval = F05Evaluator(list(val_ids), {e: ground_truth.get(e) for e in val_ids})
            fold_point = fold_eval.evaluate_threshold(
                _entity_index(all_ids[val_mask], list(val_ids)),
                [all_candidates[i] for i in np.flatnonzero(val_mask)],
                oof_scores[val_mask],
                best.threshold,
            )
            per_fold_f05.append(fold_point.f05)
            fold_reports.append(
                {"model": name, "fold": fold_index, "n_val_entities": len(val_ids),
                 "f05": round(fold_point.f05, 6), "precision": round(fold_point.macro_precision, 6),
                 "recall": round(fold_point.macro_recall, 6),
                 "n_train_pairs": int(fold_masks[fold_index].sum()), "n_val_pairs": int(val_mask.sum())}
            )

        results[name] = ModelScore(
            name=name, point=best, f05_by_fold=per_fold_f05,
            f05_mean=float(np.mean(per_fold_f05)) if per_fold_f05 else 0.0,
            f05_std=float(np.std(per_fold_f05)) if per_fold_f05 else 0.0,
            threshold=best.threshold,
        )
        oof_store[name] = (oof_scores, valid)
        merged_importance: dict[str, float] = {}
        for model in trained:
            for key, value in model.feature_importance.items():
                merged_importance[key] = merged_importance.get(key, 0.0) + value / max(1, len(trained))
        best_model_importance = merged_importance
        LOGGER.info("Model %-24s out-of-fold F0.5=%.4f at threshold %.4f", name, best.f05, best.threshold)

    if not results:
        raise RuntimeError("No model could be cross-validated; check the data and candidate generation")

    best_model = max(results.values(), key=lambda m: (m.point.f05, m.point.macro_precision))
    LOGGER.info("Best model by out-of-fold F0.5: %s (%.4f)", best_model.name, best_model.point.f05)

    importance = [
        {"feature": k, "importance": round(v, 6)}
        for k, v in sorted(best_model_importance.items(), key=lambda kv: -abs(kv[1]))[:30]
    ]

    return ValidationReport(
        n_source1=len(universe),
        n_pool=len(candidates.pool_ids),
        folds=fold_reports,
        candidate_recall=recall.as_row(),
        model_scores=[results[m].as_row() for m in model_names if m in results],
        best_model=best_model.name,
        best_threshold=best_model.threshold,
        best_f05=best_model.point.f05,
        best_point=best_model.point.as_row(),
        per_fold_f05=best_model.f05_by_fold,
        feature_importance=importance,
        blocking_stats=candidates.stats,
        runtime_seconds={"cross_validation": round(time.perf_counter() - t_start, 2)},
    )


def reference_f05(
    result, ground_truth: GroundTruth, threshold: float, source1_ids: Sequence[str]
) -> float:
    """Recompute F0.5 with the slow reference implementation (verification)."""
    predictions = result.predictions
    return f05_score({e: ground_truth.get(e) for e in source1_ids}, predictions)
