"""Build the disk-backed retrieval indexes for a split.

    python scripts/build_index.py --split train
    python scripts/build_index.py --split train --max-df 2000

Reads the normalised pool once per index, streams rather than materialising, and
writes flat mmap-able ``.npy`` arrays plus JSON manifests.  Peak memory is
dominated by the postings array, which is why the build is two passes (document
frequency, then postings) instead of one.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.scale.config import ScaleConfig  # noqa: E402
from src.scale.indexes import IndexBundle, build_index_bundle  # noqa: E402
from src.scale.progress import log_stage, make_guard  # noqa: E402
from src.scale.store import PoolStore  # noqa: E402

LOGGER = logging.getLogger("build_index")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=("train", "test"), required=True)
    parser.add_argument("--store-dir", default=None, help="defaults to <work>/<split>/store")
    parser.add_argument("--index-dir", default=None, help="defaults to <work>/<split>/index")
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--min-df", type=int, default=None)
    parser.add_argument("--max-df", type=int, default=None)
    parser.add_argument("--exact-key-chars", type=int, default=None)
    parser.add_argument("--memory-budget-gb", type=float, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    config = ScaleConfig.load(args.config)
    if args.work_dir:
        config.paths.work_dir = args.work_dir
    for flag, name in (
        (args.min_df, "min_df"),
        (args.max_df, "max_df"),
        (args.exact_key_chars, "exact_key_chars"),
    ):
        if flag is not None:
            setattr(config, name, flag)
    if args.memory_budget_gb:
        config.memory_budget_bytes = int(args.memory_budget_gb * 1024**3)
    config.validate()

    store_dir = Path(args.store_dir) if args.store_dir else config.paths.store_dir(args.split)
    index_dir = Path(args.index_dir) if args.index_dir else config.paths.index_dir(args.split)
    index_dir.mkdir(parents=True, exist_ok=True)
    guard = make_guard(config)

    pool = PoolStore(store_dir, config)
    LOGGER.info("%s", pool.describe())
    with log_stage(f"build_index {args.split}", guard):
        bundle = build_index_bundle(
            pool,
            n_pool=pool.n_pool,
            s2_rows=pool.s2_rows,
            config=config,
            guard=guard,
        )
    LOGGER.info("\n%s", bundle.describe())
    manifest = bundle.save(index_dir)
    LOGGER.info("wrote %s", manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
