"""The training pipeline.

Stages
------
1. load ``dataset/train`` + the ground truth
2. preprocess and retrieve candidates for every training Source 1 entity
3. measure **candidate recall** -- the hard ceiling on achievable F0.5
4. cross-validate the candidate models and pick one on out-of-fold F0.5
5. refit the winner on all candidate pairs
6. attribute the remaining errors to blocking vs. the model
7. persist the model, the chosen threshold and the reports

The selected threshold comes from out-of-fold predictions only, so the number
quoted in the report is not a training-set fit.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import pandas as pd

from ..config import ID_COLUMN, Config
from ..data.ground_truth import GroundTruth, load_ground_truth
from ..data.loader import load_train
from ..evaluation.error_analysis import analyse_errors
from ..evaluation.validation import cross_validate
from ..models.predict import MatchingResult, build_candidate_scores_frame, decide_matches
from ..models.train import build_label_matrix, train_model
from .stages import RetrievalResult, ScoredCandidates, build_retrieval, score_candidates

LOGGER = logging.getLogger(__name__)

MODEL_BUNDLE = "model.joblib"
TRAINING_REPORT = "training_report.json"
ERROR_REPORT = "error_analysis.json"
THRESHOLD_REPORT = "threshold_sweep.txt"


@dataclass
class TrainingOutcome:
    """Everything the training run produced."""

    model_name: str
    threshold: float
    oof_f05: float
    candidate_recall: float
    model: object
    retrieval: RetrievalResult
    features: ScoredCandidates
    result: MatchingResult
    reports: dict[str, Path] = field(default_factory=dict)

    def summary(self) -> str:
        return "\n".join(
            [
                "TRAINING COMPLETE",
                "-" * 70,
                f"  selected model        : {self.model_name}",
                f"  selected threshold    : {self.threshold:.4f}",
                f"  out-of-fold F0.5      : {self.oof_f05:.4f}",
                f"  candidate recall      : {self.candidate_recall:.4f}  (ceiling on F0.5)",
                f"  training pairs        : {self.features.features.X.shape[0]}",
                f"  matched pairs         : {self.result.n_matched_pairs}",
                f"  entities with a match : {self.result.n_entities_with_match}",
                f"  entities unmatched    : {self.result.n_entities_without_match}",
            ]
        )


def run_training(
    config: Config,
    *,
    limit_source1: int | None = None,
    skip_validation: bool = False,
    save: bool = True,
) -> TrainingOutcome:
    """Run the whole training flow and return the fitted artefacts."""
    started = time.perf_counter()
    raw = load_train(config.paths.train_dir)
    source1_ids = [str(x) for x in raw.source1[ID_COLUMN].tolist()]
    if limit_source1:
        raw.source1 = raw.source1.head(limit_source1).copy()
        source1_ids = [str(x) for x in raw.source1[ID_COLUMN].tolist()]
        LOGGER.warning("Training on the first %d Source 1 entities only (debug mode)", limit_source1)

    ground_truth = load_ground_truth(config.paths.train_dir, source1_ids)
    LOGGER.info(
        "Ground truth: %d true pairs over %d entities",
        sum(len(v) for v in ground_truth.matches.values()),
        sum(1 for v in ground_truth.matches.values() if v),
    )

    retrieval = build_retrieval(raw, config)
    features = score_candidates(retrieval, config, None)

    threshold = config.threshold
    best_name = config.model.candidates[0] if config.model.candidates else "logistic_regression"
    oof_f05 = float("nan")
    candidate_recall = float("nan")
    validation_report = None

    if not skip_validation:
        validation_report = cross_validate(features.features, retrieval.candidates, ground_truth, config)
        best = validation_report
        best_name = best.best_model
        threshold = best.best_threshold
        oof_f05 = best.best_f05
        candidate_recall = float(best.candidate_recall.get("candidate_recall") or 0.0)
    else:
        LOGGER.warning("Skipping cross-validation; the threshold is NOT data-driven")

    # --- refit the winner on every candidate pair --------------------------
    y = build_label_matrix(
        [str(s) for s in features.features.source1_ids],
        [str(c) for c in features.features.candidate_ids],
        ground_truth.matches,
    )
    model = train_model(
        best_name, features.features.X, y, features.features.columns, config.model,
    )

    train_scores = model.predict_proba(features.features.X)
    result = decide_matches(
        [str(s) for s in features.features.source1_ids],
        [str(c) for c in features.features.candidate_ids],
        train_scores,
        threshold,
        all_source1_ids=retrieval.source1_ids,
    )
    LOGGER.info("In-sample F0.5 (optimistic, for reference only): see error_analysis.json")

    outcome = TrainingOutcome(
        model_name=best_name,
        threshold=threshold,
        oof_f05=oof_f05,
        candidate_recall=candidate_recall,
        model=model,
        retrieval=retrieval,
        features=features,
        result=result,
    )

    # --- error attribution -------------------------------------------------
    analysis = analyse_errors(
        retrieval.candidates,
        result,
        ground_truth,
        train_scores,
        y,
        retrieval.source1_ids,
        _record_lookup(raw.source1),
        _record_lookup(
            pd.concat([raw.source2, raw.source3], ignore_index=True)
        ),
        pd.DataFrame(features.features.X, columns=features.features.columns),
        features.features.columns,
        threshold=threshold,
    )
    print("\n" + analysis.render())

    if save:
        outcome.reports = _persist(outcome, analysis, config, validation_report, started)
    return outcome


def _record_lookup(frame: pd.DataFrame) -> dict[str, pd.Series]:
    return {str(row[ID_COLUMN]): row for _, row in frame.iterrows()}


def _persist(
    outcome: TrainingOutcome,
    analysis,
    config: Config,
    validation_report,
    started: float,
) -> dict[str, Path]:
    """Write the model bundle and every report; return the written paths."""
    models_dir = Path(config.paths.models_dir)
    reports_dir = Path(config.paths.reports_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    bundle_path = models_dir / MODEL_BUNDLE
    joblib.dump(
        {
            "model": outcome.model.estimator,
            "model_name": outcome.model_name,
            "feature_names": outcome.features.features.columns,
            "threshold": outcome.threshold,
            "oof_f05": outcome.oof_f05,
            "candidate_recall": outcome.candidate_recall,
            "config": config.to_dict(),
        },
        bundle_path,
    )
    LOGGER.info("Saved model bundle to %s", bundle_path)

    written = {"model": bundle_path}

    analysis_path = reports_dir / ERROR_REPORT
    analysis.save(analysis_path)
    written["error_analysis"] = analysis_path

    summary = {
        "model_name": outcome.model_name,
        "threshold": outcome.threshold,
        "out_of_fold_f05": outcome.oof_f05,
        "candidate_recall": outcome.candidate_recall,
        "n_source1": len(outcome.retrieval.source1_ids),
        "n_pool": len(outcome.retrieval.pool),
        "n_candidate_pairs": int(outcome.features.features.X.shape[0]),
        "n_matched_pairs": outcome.result.n_matched_pairs,
        "n_entities_with_match": outcome.result.n_entities_with_match,
        "n_entities_without_match": outcome.result.n_entities_without_match,
        "blocking_stats": outcome.retrieval.candidates.stats,
        "runtime_seconds": round(time.perf_counter() - started, 2),
        "validation": validation_report.as_row() if validation_report is not None else None,
    }
    report_path = reports_dir / TRAINING_REPORT
    report_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    written["training_report"] = report_path

    scores_path = reports_dir / "training_candidate_scores.tsv"
    build_candidate_scores_frame(
        [str(s) for s in outcome.features.features.source1_ids],
        [str(c) for c in outcome.features.features.candidate_ids],
        train_scores,
        [str(s) for s in outcome.features.features.candidate_sources],
    ).to_csv(scores_path, sep="\t", index=False, lineterminator="\n")
    written["candidate_scores"] = scores_path
    return written
