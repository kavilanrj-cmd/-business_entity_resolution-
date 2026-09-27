"""Smoke-test candidate generation on the first N Source 1 rows.

    python scripts/smoke_candidates.py --split train --limit 1000

Reports, per the brief: candidates for each of the first N Source 1 rows, the
maximum and average, wall-clock runtime, and peak RSS, broken down by retrieval
strategy.  With ``--train`` it also reports blocking recall against the verified
Source 1 -> Source 2 / Source 3 pairs, so the numbers mean something rather than
just being a throughput measurement.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from src.scale.candidates import CandidateGenerator, CandidateWriter, generate_candidates  # noqa: E402
from src.scale.config import ScaleConfig  # noqa: E402
from src.scale.indexes import IndexBundle  # noqa: E402
from src.scale.progress import format_bytes, make_guard  # noqa: E402
from src.scale.recall import measure_recall  # noqa: E402
from src.scale.store import PoolStore, RecordStore  # noqa: E402

LOGGER = logging.getLogger("smoke_candidates")


def _make_collector(retrieved: dict[int, set[int]]):
    """Build an ``on_batch`` callback that records each query's pool rowids."""

    def collect(rowid0: int, batch) -> None:
        for slot in range(batch.n_queries):
            mask = batch.query == slot
            retrieved[rowid0 + int(slot)] = set(batch.pool_rowid[mask].tolist())

    return collect


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument("--limit", type=int, default=1000, help="number of Source 1 rows")
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--store-dir", default=None)
    parser.add_argument("--index-dir", default=None)
    parser.add_argument("--out-dir", default=None, help="defaults to <work>/<split>/candidates")
    parser.add_argument("--config", default=None)
    parser.add_argument("--batch-rows", type=int, default=None)
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--write", action="store_true", help="also write candidate Parquet shards")
    parser.add_argument("--json", default=None, help="write the report to this JSON file")
    parser.add_argument("--show", type=int, default=0, help="print this many example queries")
    parser.add_argument(
        "--truth",
        default=None,
        help="verified-pairs TSV; enables blocking recall when given",
    )
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
    if args.batch_rows:
        config.batch_rows = args.batch_rows
    if args.max_candidates:
        config.max_candidates_per_s1 = args.max_candidates
    config.validate()

    store_dir = Path(args.store_dir) if args.store_dir else config.paths.store_dir(args.split)
    index_dir = Path(args.index_dir) if args.index_dir else config.paths.index_dir(args.split)
    out_dir = Path(args.out_dir) if args.out_dir else config.paths.candidate_dir(args.split)
    guard = make_guard(config)

    s1 = RecordStore(store_dir, "s1", config)
    pool = PoolStore(store_dir, config)
    indexes = IndexBundle.load(index_dir, mmap=True)
    LOGGER.info("s1 rows: %d | pool rows: %d", len(s1), pool.n_pool)

    writer = CandidateWriter(out_dir, config) if args.write else None
    limit = min(int(args.limit), len(s1))
    LOGGER.info("generating candidates for Source 1 rows [0:%d]", limit)

    # Collect each query's pool rowids so recall can be measured against exactly
    # the candidates this run produced, with no second generation pass.
    retrieved: dict[int, set[int]] = {}
    collector = None
    if args.truth:
        if indexes.pool_ids is None:
            raise SystemExit("--truth needs a pool id resolver; rebuild the index with ids")
        collector = _make_collector(retrieved)

    began = time.perf_counter()
    stats = generate_candidates(
        s1,
        pool,
        indexes,
        config,
        start=0,
        stop=limit,
        guard=guard,
        writer=writer,
        on_batch=collector,
        label=f"smoke {args.split}",
    )
    wall = time.perf_counter() - began
    if writer is not None:
        writer.close()
        LOGGER.info("wrote %s", writer.describe())

    # Placed after ``**stats.as_dict()`` on purpose: the guard is the authority on
    # peak RSS, and the spread above would otherwise silently win.
    peak = max(guard.peak(), guard.observe(), stats.peak_bytes)
    report: dict = {
        "split": args.split,
        "s1_rows_examined": limit,
        "pool_rows": pool.n_pool,
        "index_terms": indexes.name.n_terms,
        "index_postings": indexes.name.n_postings,
        "index_bytes": indexes.nbytes(),
        "batch_rows": config.batch_rows,
        "max_candidates_per_s1": config.max_candidates_per_s1,
        "wall_seconds": round(wall, 2),
        **stats.as_dict(),
        "peak_rss_bytes": peak,
        "peak_rss": format_bytes(peak),
    }

    if args.show:
        report["examples"] = _examples(s1, pool, indexes, config, limit, args.show)

    if args.truth:
        ids_frame = s1.read_range(0, limit, ["entity_id"])
        recall = measure_recall(
            args.truth,
            ids_frame["entity_id"].to_numpy(dtype=object),
            np.arange(limit, dtype=np.int64),
            retrieved,
            indexes.pool_ids,
            config=config,
        )
        report.update(recall.as_dict())
        report["recall_unmatched_examples"] = recall.unmatched_examples
        LOGGER.info("blocking recall: %s", recall.describe())

    print()
    print("=" * 72)
    print(f"candidate smoke test -- split={args.split}  first {limit:,} Source 1 rows")
    print("=" * 72)
    for key in (
        "candidates",
        "candidates_per_query_avg",
        "candidates_per_query_max",
        "empty_queries",
        "empty_query_pct",
        "wall_seconds",
        "queries_per_second",
        "candidates_per_second",
        "peak_rss",
    ):
        print(f"  {key:26s} {report.get(key)}")
    print("  by_strategy:")
    for name, count in sorted(report["by_strategy"].items(), key=lambda kv: -kv[1]):
        share = 100.0 * count / max(1, report["candidates"])
        print(f"    {name:24s} {count:>12,}  {share:5.1f}%")
    if "examples" in report:
        print("  examples:")
        for line in report["examples"]:
            print(f"    {line}")
    if "recall_query_recall" in report:
        print("  blocking recall (verified pairs):")
        for key in (
            "recall_queries_labelled",
            "recall_queries_unlabelled",
            "recall_verified_pairs",
            "recall_pairs_retrieved",
            "recall_query_recall",
            "recall_pair_recall",
            "recall_all_pairs_recall",
            "recall_pool_ids_missing",
        ):
            print(f"    {key:28s} {report.get(key)}")
        for item in report.get("recall_unmatched_examples", [])[:3]:
            print(f"    missed {item['retrieved']}/{item['of']} for {item['s1_id']}")
    print("=" * 72)

    if args.json:
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        LOGGER.info("wrote %s", path)
    return 0


def _examples(s1, pool, indexes, config, limit: int, count: int) -> list[str]:
    """Show the first few Source 1 queries and what each retrieved."""
    generator = CandidateGenerator(s1, pool, indexes, config)
    lines: list[str] = []
    remaining = count
    for rowid0, batch in generator.batches(start=0, stop=limit):
        if remaining <= 0 or len(batch) == 0:
            break
        names = s1.read_range(rowid0, rowid0 + batch.n_queries, ["name_core", "country"])
        for slot in range(min(remaining, batch.n_queries)):
            mask = batch.query == slot
            found = batch.pool_rowid[mask]
            scored = batch.content_score[mask]
            if found.size:
                pool_names = pool.read_where(found[:4], ["name_core"])["name_core"].tolist()
            else:
                pool_names = []
            top = ", ".join(
                f"{n!r}@{s:.2f}" for n, s in zip(pool_names, scored[: len(pool_names)].tolist())
            )
            strategies = batch.named_strategies(int(np.flatnonzero(mask)[0])) if found.size else []
            lines.append(
                f"[{rowid0 + slot}] {names['name_core'].iloc[slot]!r} "
                f"({names['country'].iloc[slot]}) -> {int(found.size)} via "
                f"{'+'.join(strategies)}: {top}"
            )
            remaining -= 1
        if remaining <= 0:
            break
    return lines


if __name__ == "__main__":
    raise SystemExit(main())
