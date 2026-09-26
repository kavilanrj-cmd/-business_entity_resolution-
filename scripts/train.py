"""Train the model and write the reports.

    python -m scripts.train
    python -m scripts.train --skip-validation          # fast, threshold is not data-driven
    python -m scripts.train --limit-source1 300        # quick smoke run
    python -m scripts.train --config config.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow `python scripts/train.py` as well as `python -m scripts.train`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config, setup_logging  # noqa: E402
from src.pipeline.train_pipeline import run_training  # noqa: E402

LOGGER = logging.getLogger("scripts.train")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the entity resolution model.")
    parser.add_argument("--config", default=None, help="optional YAML configuration file")
    parser.add_argument("--base-dir", default=None, help="project root (defaults to the CWD)")
    parser.add_argument("--train-dir", default=None, help="override the training data directory")
    parser.add_argument("--models-dir", default=None, help="where to write the model bundle")
    parser.add_argument("--reports-dir", default=None, help="where to write the reports")
    parser.add_argument("--models", nargs="+", default=None,
                        help="candidate models to compare, e.g. logistic_regression random_forest")
    parser.add_argument("--n-splits", type=int, default=None, help="cross-validation folds")
    parser.add_argument("--limit-source1", type=int, default=None,
                        help="only use the first N Source 1 entities (debugging)")
    parser.add_argument("--skip-validation", action="store_true",
                        help="skip cross-validation (the threshold will not be data-driven)")
    parser.add_argument("--no-save", action="store_true", help="do not write the model or reports")
    parser.add_argument("--log-level", default=None, help="DEBUG / INFO / WARNING")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = Config.load(args.config, overrides={
        k: v for k, v in {
            "paths.base_dir": args.base_dir,
            "paths.train_dir": args.train_dir,
            "paths.models_dir": args.models_dir,
            "paths.reports_dir": args.reports_dir,
            "model.candidates": tuple(args.models) if args.models else None,
            "model.n_splits": args.n_splits,
            "log_level": args.log_level,
        }.items() if v is not None
    })
    config.ensure_dirs()
    setup_logging(config.log_level)

    LOGGER.info("Training with %d candidate model(s)", len(config.model.candidates))
    outcome = run_training(
        config,
        limit_source1=args.limit_source1,
        skip_validation=args.skip_validation,
        save=not args.no_save,
    )
    print("\n" + outcome.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
