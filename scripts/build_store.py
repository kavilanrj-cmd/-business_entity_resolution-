"""Normalise a split's TSVs into sharded Parquet.

    python scripts/build_store.py --split train
    python scripts/build_store.py --split test --limit-s1 1000

Reads each TSV once in bounded chunks, normalises names and addresses, and
writes fixed-size Parquet shards plus a manifest.  The original TSVs are never
opened for writing.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.scale.config import ScaleConfig  # noqa: E402
from src.scale.progress import log_stage, make_guard  # noqa: E402
from src.scale.store import (  # noqa: E402
    SOURCE_FILES,
    SplitStore,
    build_table,
    write_store_manifest,
)

LOGGER = logging.getLogger("build_store")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=("train", "test"), required=True)
    parser.add_argument("--tsv-dir", default=None, help="defaults to the first known layout that has the files")
    parser.add_argument("--work-dir", default=None, help="defaults to <project>/work")
    parser.add_argument("--config", default=None, help="JSON config written by build_store --dump-config")
    parser.add_argument("--limit-s1", type=int, default=None, help="only the first N Source 1 rows")
    parser.add_argument("--limit-s2", type=int, default=None)
    parser.add_argument("--limit-s3", type=int, default=None)
    parser.add_argument("--skip-s1", action="store_true", help="build only the blocking pool")
    parser.add_argument("--shard-rows", type=int, default=None)
    parser.add_argument("--read-chunk-rows", type=int, default=None)
    parser.add_argument("--batch-rows", type=int, default=None)
    parser.add_argument("--memory-budget-gb", type=float, default=None)
    return parser.parse_args(argv)


def _resolve_tsv_dir(split: str, explicit: str | None) -> Path:
    """Find the directory holding this split's TSVs.

    The repo has two layouts -- the small ``dataset/train`` fixture and the real
    ``dataset/student_resource/dataset/train`` download -- so rather than guess,
    probe the known locations and take the first that actually has the files.
    """
    if explicit:
        return Path(explicit)
    project = Path(__file__).resolve().parents[1]
    candidates = (
        project / "dataset" / "student_resource" / "dataset" / split,
        project / "dataset" / split / "dataset",
        project / "dataset" / split,
    )
    for candidate in candidates:
        if all((candidate / name).exists() for name in SOURCE_FILES.values()):
            return candidate
    return candidates[0]


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
        (args.shard_rows, "shard_rows"),
        (args.read_chunk_rows, "read_chunk_rows"),
        (args.batch_rows, "batch_rows"),
    ):
        if flag:
            setattr(config, name, flag)
    if args.memory_budget_gb:
        config.memory_budget_bytes = int(args.memory_budget_gb * 1024**3)
    config.validate()

    tsv_dir = _resolve_tsv_dir(args.split, args.tsv_dir)
    store_dir = config.paths.store_dir(args.split)
    store_dir.mkdir(parents=True, exist_ok=True)
    guard = make_guard(config)

    limits = {"s1": args.limit_s1, "s2": args.limit_s2, "s3": args.limit_s3}
    sources = ("s2", "s3") if args.skip_s1 else ("s1", "s2", "s3")
    stats = {}
    for source in sources:
        tsv = tsv_dir / SOURCE_FILES[source]
        if not tsv.exists():
            raise FileNotFoundError(f"{tsv} not found; pass --tsv-dir")
        with log_stage(f"build_store {args.split}/{source}", guard):
            stats[source] = build_table(
                tsv,
                store_dir / source,
                source,
                config,
                guard=guard,
                limit_rows=limits[source],
            )

    payload = {
        "split": args.split,
        "tsv_dir": str(tsv_dir),
        "store_dir": str(store_dir),
        "config": config.to_dict(),
        "sources": {
            source: {
                **s.as_dict(),
                "input_bytes": s.input_bytes,
                "output_bytes": s.output_bytes,
                "file_bytes": s.file_bytes,
            }
            for source, s in stats.items()
        },
    }
    if "s2" in stats and "s3" in stats:
        payload["n_pool"] = stats["s2"].rows + stats["s3"].rows
    manifest = write_store_manifest(store_dir, payload)
    LOGGER.info("wrote %s", manifest)
    if "s1" in stats and "s2" in stats and "s3" in stats:
        LOGGER.info("\n%s", SplitStore(store_dir, config).describe())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
