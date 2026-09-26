"""The inference pipeline.

Loads the persisted model bundle, retrieves candidates for the test Source 1
entities, scores them, applies the threshold chosen during training, writes
``matching_results.tsv`` / ``candidate_pairs.tsv`` and then **validates the
result before returning**.  A run that cannot produce a well-formed submission
raises rather than leaving a broken file behind.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import pandas as pd

from ..config import ID_COLUMN, Config
from ..data.loader import load_test
from ..models.predict import MatchingResult, build_candidate_scores_frame, decide_matches
from ..models.train import TrainedModel
from ..output.candidate_pairs import build_candidate_pairs_frame
from ..output.matching_results import write_matching_results
from ..output.validate_submission import (
    ValidationOutcome,
    find_official_validator,
    validate_submission,
)
from ..output.candidate_pairs import write_candidate_pairs
from .stages import RetrievalResult, ScoredCandidates, build_retrieval, score_candidates

LOGGER = logging.getLogger(__name__)


@dataclass
class InferenceOutcome:
    """Predictions, written files and the validation verdict."""

    model_name: str
    threshold: float
    result: MatchingResult
    retrieval: RetrievalResult
    features: ScoredCandidates
    written: dict[str, Path] = field(default_factory=dict)
    validation: ValidationOutcome | None = None

    def summary(self) -> str:
        return "\n".join(
            [
                "INFERENCE COMPLETE",
                "-" * 70,
                f"  model                   : {self.model_name}",
                f"  threshold               : {self.threshold:.4f}",
                f"  test Source 1 entities  : {len(self.retrieval.source1_ids)}",
                f"  candidate pairs         : {int(self.features.features.X.shape[0])}",
                f"  matched pairs           : {self.result.n_matched_pairs}",
                f"  entities with a match   : {self.result.n_entities_with_match}",
                f"  entities unmatched      : {self.result.n_entities_without_match}",
            ]
            + ([f"  validation              : {'PASS' if self.validation.ok else 'FAIL'}"]
               if self.validation else [])
        )


def load_bundle(models_dir: str | Path) -> dict:
    """Load the bundle written by the training pipeline."""
    path = Path(models_dir) / "model.joblib"
    if not path.exists():
        raise FileNotFoundError(
            f"No trained model at {path}. Run `python -m scripts.train` first."
        )
    return joblib.load(path)


def run_inference(
    config: Config,
    *,
    test_dir: str | Path | None = None,
    models_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    limit_source1: int | None = None,
    write: bool = True,
    validate: bool = True,
) -> InferenceOutcome:
    """Score the test set and emit a validated submission."""
    started = time.perf_counter()
    bundle = load_bundle(models_dir or config.paths.models_dir)
    model = TrainedModel(
        name=bundle["model_name"],
        estimator=bundle["model"],
        feature_names=list(bundle["feature_names"]),
        n_train_pairs=0,
        n_train_positives=0,
        train_seconds=0.0,
    )
    threshold = float(bundle["threshold"])
    LOGGER.info("Loaded '%s' with threshold %.4f", model.name, threshold)

    raw = load_test(test_dir or config.paths.test_dir)
    if limit_source1:
        raw.source1 = raw.source1.head(limit_source1).copy()
        LOGGER.warning("Scoring the first %d test Source 1 entities only (debug mode)", limit_source1)

    retrieval = build_retrieval(raw, config)
    expected = list(bundle["feature_names"])
    features = score_candidates(retrieval, config, model)
    if features.features.columns != expected:
        raise ValueError(
            "Feature layout does not match the trained model.\n"
            f"  trained : {expected}\n  current : {features.features.columns}\n"
            "Retrain after changing the feature configuration."
        )

    s1_ids = [str(s) for s in features.features.source1_ids]
    cand_ids = [str(c) for c in features.features.candidate_ids]
    result = decide_matches(s1_ids, cand_ids, features.scores, threshold, all_source1_ids=retrieval.source1_ids)

    outcome = InferenceOutcome(
        model_name=model.name, threshold=threshold, result=result,
        retrieval=retrieval, features=features,
    )

    out_dir = Path(output_dir or config.paths.output_dir)
    if write:
        matching = result.as_frame(retrieval.source1_ids, config.output)
        candidate_frame = build_candidate_pairs_frame(
            retrieval.source1_ids, retrieval.candidates_by_entity(), config.output
        )
        outcome.written = {
            "matching_results": write_matching_results(matching, out_dir, config.output),
            "candidate_pairs": write_candidate_pairs(candidate_frame, out_dir, config.output),
        }

    if validate:
        outcome.validation = validate_submission(
            outcome.written.get("matching_results", out_dir / config.output.matching_results_file),
            outcome.written.get("candidate_pairs"),
            raw.source1, raw.source2, raw.source3,
            official_validator=find_official_validator(config.paths.base_dir),
        )
        print("\n" + outcome.validation.render())
        outcome.validation.raise_if_failed()

    LOGGER.info("Inference finished in %.1fs", time.perf_counter() - started)
    return outcome
