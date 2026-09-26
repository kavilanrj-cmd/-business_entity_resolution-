"""Score the test set and write a validated submission.

    python -m scripts.infer
    python -m scripts.infer --limit-source1 100     # quick smoke run
    python -m scripts.infer --no-validate
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config, setup_logging  # noqa: E402
from src.pipeline.inference_pipeline import run_inference  # noqa: E402

LOGGER = logging.getLogger("scripts.infer")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the submission from a trained model.")
    parser.add_argument("--config", default=None, help="optional YAML configuration file")
    parser.add_argument("--base-dir", default=None, help="project root (defaults to the CWD)")
    parser.add_argument("--test-dir", default=None, help="override the test data directory")
    parser.add_argument("--models-dir", default=None, help="where the trained bundle lives")
    parser.add_argument("--output-dir", default=None, help="where to write the submission")
    parser.add_argument("--limit-source1", type=int, default=None,
                        help="only score the first N test Source 1 entities (debugging)")
    parser.add_argument("--no-write", action="store_true", help="score without writing files")
    parser.add_argument("--no-validate", action="store_true", help="skip submission validation")
    parser.add_argument("--log-level", default=None, help="DEBUG / INFO / WARNING")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = Config.load(args.config, overrides={
        k: v for k, v in {
            "paths.base_dir": args.base_dir,
            "paths.test_dir": args.test_dir,
            "paths.models_dir": args.models_dir,
            "paths.output_dir": args.output_dir,
            "log_level": args.log_level,
        }.items() if v is not None
    })
    config.ensure_dirs()
    setup_logging(config.log_level)

    outcome = run_inference(
        config,
        test_dir=args.test_dir,
        models_dir=args.models_dir,
        output_dir=args.output_dir,
        limit_source1=args.limit_source1,
        write=not args.no_write,
        validate=not args.no_validate,
    )
    print("\n" + outcome.summary())
    return 0 if (outcome.validation is None or outcome.validation.ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
