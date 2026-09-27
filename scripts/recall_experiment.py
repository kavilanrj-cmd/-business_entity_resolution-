"""Controlled candidate-generation experiment on a fixed Source 1 sample.

    python scripts/recall_experiment.py --split train --limit 1000

Answers two questions with measurements instead of assumptions:

1. What does raising the candidate cap actually buy?  A cap sweep over the same
   rows, with recall, volume, percentiles, runtime and peak RSS for each step.
2. Does a single global rank waste slots?  Per-strategy attribution: how many
   candidates each strategy contributes, how many true matches it recovers, and
   -- the number that decides whether a strategy earns its slots -- how many true
   matches *only* that strategy finds.

Method
------
The cap is the last operation in ``CandidateGenerator._merge``, after the six
strategies have been unioned and ranked.  So the whole sweep can be evaluated
from **one** uncapped generation pass: capture the ranked arrays once, then
re-slice them per configuration.  That is exact, not an approximation -- slicing
the first ``cap`` entries of a query reproduces what a capped run would emit --
and ``--self-check`` verifies it against a real capped run rather than asserting
it in a comment.

Runtime honesty: raising the cap does not make generation slower or faster,
because generation cost is set by the posting budgets and the neighbourhood
window, not by the cap.  The cap's real cost lands downstream, in feature
extraction and training, and scales with candidate volume.  Both numbers are
reported, and the per-configuration runtime column is the selection step.
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

from src.scale.candidates import (  # noqa: E402
    EXACT_STRATEGIES,
    STRATEGY_WEIGHT,
    CandidateGenerator,
    _rank_within_group,
    select_with_quotas,
)
from src.scale.config import STRATEGY_BITS, STRATEGY_ORDER, ScaleConfig  # noqa: E402
from src.scale.indexes import IndexBundle  # noqa: E402
from src.scale.progress import format_bytes, make_guard, rss_bytes  # noqa: E402
from src.scale.recall import load_verified_pairs  # noqa: E402
from src.scale.store import PoolStore, RecordStore  # noqa: E402

LOGGER = logging.getLogger("recall_experiment")

#: Human-facing grouping: the brief asks for "exact name", "name token",
#: "character/ngram", "address", "country" and "any other".
STRATEGY_LABEL = {
    "exact_normalized": "exact name (normalized)",
    "exact_core": "exact name (legal-form stripped)",
    "name_token": "name token",
    "sorted_neighbourhood": "character/ngram (sorted neighbourhood)",
    "address_token": "address token",
    "country_token": "country",
}

CAP_SWEEP = (50, 100, 150, 200, 300, 500)


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def _percentiles(counts: np.ndarray) -> dict[str, float]:
    if counts.size == 0:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    return {
        "mean": round(float(counts.mean()), 2),
        "p50": float(np.percentile(counts, 50)),
        "p95": float(np.percentile(counts, 95)),
        "p99": float(np.percentile(counts, 99)),
        "max": int(counts.max()),
    }


def _recall(
    query: np.ndarray,
    pool_rowid: np.ndarray,
    truth: dict[int, set[int]],
    n_queries: int,
) -> dict:
    """Query / pair / all-pairs recall for a selected candidate set."""
    found: dict[int, set[int]] = {i: set() for i in range(n_queries)}
    for q, p in zip(query.tolist(), pool_rowid.tolist()):
        found[q].add(p)

    labelled = 0
    unlabelled = 0
    pairs = 0
    pairs_found = 0
    with_any = 0
    with_all = 0
    for rowid, want in truth.items():
        if not want:
            unlabelled += 1
            continue
        labelled += 1
        hits = len(want & found.get(rowid, ()))
        pairs += len(want)
        pairs_found += hits
        if hits:
            with_any += 1
        if hits == len(want):
            with_all += 1
    return {
        "queries": n_queries,
        "queries_labelled": labelled,
        "queries_unlabelled": unlabelled,
        "verified_pairs": pairs,
        "pairs_retrieved": pairs_found,
        "query_recall": round(with_any / labelled, 4) if labelled else None,
        "pair_recall": round(pairs_found / pairs, 4) if pairs else None,
        "all_pairs_recall": round(with_all / labelled, 4) if labelled else None,
        "queries_with_no_match": labelled - with_any,
    }


def _strategy_attribution(
    query: np.ndarray,
    pool_rowid: np.ndarray,
    mask: np.ndarray,
    truth: dict[int, set[int]],
) -> list[dict]:
    """Per-strategy volume and true-match yield, over the full uncapped union.

    "Recovered" counts a verified pair once per strategy that found it, so the
    column can sum above the total.  "Only this strategy" is the exclusive
    contribution: if that strategy's hits were dropped, the pair would be lost.
    """
    rows = [
        {
            "strategy": strategy,
            "label": STRATEGY_LABEL[strategy],
            "weight": STRATEGY_WEIGHT[strategy],
            "candidates": int(np.count_nonzero(mask & STRATEGY_BITS[strategy])),
        }
        for strategy in STRATEGY_ORDER
    ]
    if not truth:
        return rows

    # One sorted pass over the union, then one searchsorted per strategy, instead
    # of a Python loop over verified pairs x strategies.
    order = np.lexsort((pool_rowid, query))
    flat = query[order].astype(np.int64) * np.int64(1 << 32) + pool_rowid[order].astype(np.int64)
    sorted_mask = mask[order]

    wanted = sorted({(slot, pool) for slot, pools in truth.items() for pool in pools})
    want_flat = np.array([q * (1 << 32) + p for q, p in wanted], dtype=np.int64)
    pos = np.searchsorted(flat, want_flat)
    clipped = np.clip(pos, 0, flat.size - 1)
    present = (pos < flat.size) & (flat[clipped] == want_flat)
    # present should be all True: every verified pair is in the uncapped union.
    other = present.copy()
    # ``other`` starts as True so a strategy that recovers a pair can still be
    # its exclusive finder; exclusive is computed as recovered & ~any_other.
    any_other = present.copy()
    for row in rows:
        bit = STRATEGY_BITS[row["strategy"]]
        got = present & ((sorted_mask[clipped] & bit) != 0)
        recovered = int(got.sum())
        # Exclusive: found here, and not found by any *other* strategy.
        others = np.zeros_like(got)
        for other_row in rows:
            if other_row["strategy"] == row["strategy"]:
                continue
            others |= (sorted_mask[clipped] & STRATEGY_BITS[other_row["strategy"]]) != 0
        exclusive = int((got & ~others).sum())
        row["pairs_recovered"] = recovered
        row["pairs_recovered_only_here"] = exclusive
        row["pair_recall_if_removed"] = round((len(wanted) - exclusive) / len(wanted), 4)
    del any_other, other
    total_candidates = sum(row["candidates"] for row in rows) or 1
    for row in rows:
        row["share_of_candidates"] = round(100.0 * row["candidates"] / total_candidates, 1)
    return rows


def _rank_diagnostics(
    query: np.ndarray,
    pool_rowid: np.ndarray,
    mask: np.ndarray,
    truth: dict[int, set[int]],
    caps: list[int],
) -> list[dict]:
    """Where do each strategy's exclusive true matches sit in the global rank?

    The cap sweep says the cap is the binding constraint, but not *why*.  This
    answers it: for every verified pair, which strategy (if any) is the sole
    finder, and what global rank does that pair have.  If address-only pairs sit
    far down the ranking, then a cap is cutting them off for a reason that has
    nothing to do with how well they match -- which is the difference between
    "raise the cap" and "fix the rank".
    """
    # Search in (query, pool_rowid) order so searchsorted works, but keep the
    # rank in the batch's *original* order, which is already best-first within
    # each query.  Permuting and then ranking would scramble the ranks: the
    # uncapped batch is ordered by score, not by pool rowid.
    order = np.lexsort((pool_rowid, query))
    flat = query[order].astype(np.int64) * np.int64(1 << 32) + pool_rowid[order].astype(np.int64)
    sorted_mask = mask[order]
    original_rank = _rank_within_group(query)[order]

    wanted = sorted({(slot, pool) for slot, pools in truth.items() for pool in pools})
    want_flat = np.array([q * (1 << 32) + p for q, p in wanted], dtype=np.int64)
    pos = np.clip(np.searchsorted(flat, want_flat), 0, flat.size - 1)
    present = (flat[pos] == want_flat) & (pos < flat.size)
    bits = sorted_mask[pos]

    def sole_finder(bits_row: int) -> str:
        hits = [s for s in STRATEGY_ORDER if bits_row & int(STRATEGY_BITS[s])]
        return hits[0] if len(hits) == 1 else ("none" if not hits else "multiple")

    classes = np.array([sole_finder(int(b)) for b in bits])
    pair_rank = original_rank[pos]
    rows: list[dict] = []
    for strategy in list(STRATEGY_ORDER) + ["multiple", "none"]:
        sel = present & (classes == strategy)
        total = int(sel.sum())
        if total == 0:
            continue
        this_rank = pair_rank[sel]
        entry = {
            "class": strategy if strategy in STRATEGY_ORDER else f"{strategy} strategies",
            "verified_pairs": total,
            "median_global_rank": int(np.median(this_rank)),
            "p90_global_rank": int(np.percentile(this_rank, 90)),
            "recovered_at_cap": {},
        }
        for cap in caps:
            entry["recovered_at_cap"][str(cap)] = int((this_rank < cap).sum())
        rows.append(entry)
    return rows


def _efficiency_frontier(results: list[dict]) -> list[dict]:
    """For each configuration, the cheapest plain cap that matches its recall.

    The cap sweep alone answers "how much recall does volume buy", but the
    question that matters is whether a quota configuration is *better than just
    turning the cap up*.  So for every configuration, find the smallest plain-cap
    run whose pair recall is at least as high, and report how much volume the
    configuration saves against it.  A negative saving means the configuration is
    dominated by simply raising the cap.
    """
    plain = [r for r in results if r["family"] == "cap sweep"]
    plain.sort(key=lambda r: r["candidates"])
    rows = []
    for row in results:
        if row["family"] == "cap sweep":
            continue
        target = float(row["pair_recall"])
        rivals = [p for p in plain if float(p["pair_recall"]) >= target - 1e-12]
        best = min(rivals, key=lambda p: p["candidates"]) if rivals else None
        rows.append({
            "config": row["name"],
            "candidates": row["candidates"],
            "pair_recall": row["pair_recall"],
            "all_pairs_recall": row["all_pairs_recall"],
            "cheapest_matching_plain_cap": best["cap"] if best else None,
            "cheapest_matching_candidates": best["candidates"] if best else None,
            "volume_saved_pct": (
                round(100.0 * (1 - row["candidates"] / best["candidates"]), 1)
                if best else None
            ),
        })
    rows.sort(key=lambda r: -float(r["pair_recall"]))
    return rows


# --------------------------------------------------------------------------
# configurations
# --------------------------------------------------------------------------
def build_configurations() -> list[dict]:
    """The cap sweep plus the quota and adaptive variants.

    Quotas are deliberately concrete numbers to be measured, not tuned guesses;
    Configuration D is filled in by :func:`design_adaptive` from the measured
    per-strategy yield rather than chosen by intuition.
    """
    configs: list[dict] = []
    for cap in CAP_SWEEP:
        configs.append(
            {
                "name": f"cap_{cap}",
                "family": "cap sweep",
                "cap": cap,
                "quotas": None,
                "note": f"single global rank, {cap} candidates/query",
            }
        )
    configs.append(
        {
            "name": "A_cap100",
            "family": "named",
            "cap": 100,
            "quotas": None,
            "note": "Configuration A: current production behaviour",
        }
    )
    configs.append(
        {
            "name": "B_cap200",
            "family": "named",
            "cap": 200,
            "quotas": None,
            "note": "Configuration B: double the cap, same single rank",
        }
    )
    configs.append(
        {
            "name": "C_quota_50_30_20",
            "family": "named",
            "cap": 100,
            "quotas": {
                "name_token": 50,
                "address_token": 30,
                "sorted_neighbourhood": 20,
                "country_token": 10,
            },
            "note": "Configuration C: 50 name / 30 address / 20 character / 10 country",
        }
    )
    # The attribution shows country_token recovers 0 verified pairs and is the
    # sole finder of 0, while still contributing 50,000 candidates that compete
    # for cap slots under a global rank.  E and F test dropping country-only
    # candidates -- narrower than banning the strategy, so a candidate that
    # country matched and the name index also matched keeps its name evidence.
    configs.append(
        {
            "name": "E_cap100_no_country",
            "family": "evidence",
            "cap": 100,
            "quotas": None,
            "exclude_only": ("country_token",),
            "note": "cap 100, country-only candidates dropped (measured 0 yield)",
        }
    )
    configs.append(
        {
            "name": "F_cap200_no_country",
            "family": "evidence",
            "cap": 200,
            "quotas": None,
            "exclude_only": ("country_token",),
            "note": "cap 200, country-only candidates dropped (measured 0 yield)",
        }
    )
    # G is the non-adaptive control for D: same measured quota shape for every
    # query.  If D lands on G, switching on the exact-name signal buys nothing
    # and the control is the configuration to keep.
    configs.append(
        {
            "name": "G_quota_measured_100",
            "family": "evidence",
            "cap": 100,
            "quotas": {
                "name_token": 18,
                "address_token": 56,
                "sorted_neighbourhood": 6,
            },
            "exclude_only": ("country_token", "exact_normalized"),
            "note": "measured quota (exclusive + 0.15*recovery), non-adaptive control",
        }
    )
    configs.append(
        {
            "name": "H_quota_measured_200",
            "family": "evidence",
            "cap": 200,
            "quotas": {
                "name_token": 30,
                "address_token": 100,
                "sorted_neighbourhood": 14,
            },
            "exclude_only": ("country_token", "exact_normalized"),
            "note": "measured quota scaled to cap 200",
        }
    )
    # The same quota shapes are measured again in :func:`build_union_configurations`,
    # with the cap applied to the quota union instead of to the whole batch.  In
    # "rank" mode the cap is already full of name candidates when the quotas are
    # consulted, so the quotas are inert -- which is why every rank-mode variant
    # above lands on the plain cap.
    return configs


def build_union_configurations() -> list[dict]:
    """Quota configurations with the cap applied to the quota union.

    Identical quota shapes to the "rank"-mode variants, so the only difference
    measured is *when* the cap is applied.  If union mode does not beat the
    plain cap either, then per-strategy quotas are not the lever at this cap and
    the honest recommendation is to keep the single global rank.
    """
    return [
        {
            "name": "C2_quota_50_30_20_union",
            "family": "union mode",
            "cap": 100,
            "cap_mode": "union",
            "quotas": {
                "name_token": 50,
                "address_token": 30,
                "sorted_neighbourhood": 20,
                "country_token": 10,
            },
            "note": "Configuration C quotas, cap applied to the quota union",
        },
        {
            "name": "G2_quota_measured_100_union",
            "family": "union mode",
            "cap": 100,
            "cap_mode": "union",
            "quotas": {"name_token": 18, "address_token": 56, "sorted_neighbourhood": 6},
            "exclude_only": ("country_token", "exact_normalized"),
            "note": "measured quota, cap applied to the quota union",
        },
        {
            "name": "H2_quota_measured_200_union",
            "family": "union mode",
            "cap": 200,
            "cap_mode": "union",
            "quotas": {"name_token": 30, "address_token": 100, "sorted_neighbourhood": 14},
            "exclude_only": ("country_token", "exact_normalized"),
            "note": "measured quota at cap 200, cap applied to the quota union",
        },
        {
            "name": "I2_address_heavy_200_union",
            "family": "union mode",
            "cap": 200,
            "cap_mode": "union",
            "quotas": {"name_token": 20, "address_token": 150, "sorted_neighbourhood": 20},
            "exclude_only": ("country_token", "exact_normalized"),
            "note": "address-heavy: 150 of 200 slots to the sole finder of 807 pairs",
        },
        {
            "name": "J2_address_170_200_union",
            "family": "union mode",
            "cap": 200,
            "cap_mode": "union",
            "quotas": {"name_token": 10, "address_token": 170, "sorted_neighbourhood": 20},
            "exclude_only": ("country_token", "exact_normalized"),
            "note": "address-heavier still: 170 of 200 slots to address",
        },
        {
            "name": "K2_address_120_150_union",
            "family": "union mode",
            "cap": 150,
            "cap_mode": "union",
            "quotas": {"name_token": 10, "address_token": 120, "sorted_neighbourhood": 20},
            "exclude_only": ("country_token", "exact_normalized"),
            "note": "address-heavy at cap 150, to compare against plain cap 200",
        },
        {
            "name": "L2_address_200_250_union",
            "family": "union mode",
            "cap": 250,
            "cap_mode": "union",
            "quotas": {"name_token": 20, "address_token": 200, "sorted_neighbourhood": 30},
            "exclude_only": ("country_token", "exact_normalized"),
            "note": "address-heavy at cap 250, to compare against plain cap 300",
        },
        {
            "name": "M2_name_heavy_200_union",
            "family": "union mode",
            "cap": 200,
            "cap_mode": "union",
            "quotas": {"name_token": 100, "address_token": 70, "sorted_neighbourhood": 30},
            "exclude_only": ("country_token", "exact_normalized"),
            "note": "name-heavy control: does leaning on name instead of address cost recall?",
        },
    ]


def design_adaptive(
    attribution: list[dict], cap: int, name: str = "D_adaptive", cap_mode: str = "rank"
) -> dict:
    """Configuration D: quotas computed from the measured strategy yield.

    Slot allocation under a *tight* cap is governed by exclusive yield, not by
    raw recovery.  A candidate that three strategies all found will survive
    truncation by any one of them, so spending three slots to protect it is
    waste; only a strategy that is the sole finder of a verified pair needs
    protected slots.  The measurement on this sample:

        address_token   807 pairs found by nobody else  (23.6% of all pairs)
        name_token       88
        exact_core       37
        sorted_neighbour  9
        exact_normalized  0
        country_token    0

    Allocating purely by exclusive yield would starve name_token to ~9 slots,
    which also throws away its value as corroboration for a downstream scorer.
    So each strategy is weighted ``exclusive + 0.15 * recovery``: exclusive yield
    buys protection, recovery keeps a floor under the corroborating channels.
    Zero-yield strategies are dropped entirely rather than given filler quota.
    """
    exclusive = {row["strategy"]: row.get("pairs_recovered_only_here", 0) for row in attribution}
    recovery = {row["strategy"]: row.get("pairs_recovered", 0) for row in attribution}
    scored = {s: exclusive.get(s, 0) + 0.15 * recovery.get(s, 0) for s in STRATEGY_ORDER}
    drop = tuple(s for s in STRATEGY_ORDER if not scored.get(s))
    pool = sum(scored.values())
    quotas = {s: max(4, int(round(cap * scored[s] / pool))) for s in STRATEGY_ORDER if scored.get(s)}
    # With an exact-name hit the name channel is already answered, so its slots
    # are worth less than address slots; without one, the name channel is the
    # only thing that can find the match, so it keeps them.
    with_exact = dict(quotas)
    shift = max(4, with_exact.get("address_token", 0) - with_exact.get("name_token", 0))
    with_exact["address_token"] = with_exact.get("address_token", 0) + shift
    with_exact["name_token"] = max(4, with_exact.get("name_token", 0) - shift)
    return {
        "name": name,
        "family": "named" if cap_mode == "rank" else "union mode",
        "cap": cap,
        "cap_mode": cap_mode,
        "quotas": quotas,
        "with_exact": with_exact,
        "drop_zero_yield": drop,
        "adaptive": True,
        "note": (
            "Configuration D: quota = cap * (exclusive + 0.15*recovery)/sum, "
            f"zero-yield strategies dropped ({', '.join(drop) or 'none'}), "
            f"cap_mode={cap_mode}, address favoured when an exact-name hit exists"
        ),
    }


def _select_adaptive(spec: dict, query: np.ndarray, mask: np.ndarray, n_queries: int) -> np.ndarray:
    """Apply the adaptive rule: one measured quota set for queries that have an
    exact-name hit, the base set for those that do not, then the shared cap."""
    exact = np.zeros(n_queries, dtype=bool)
    for strategy in EXACT_STRATEGIES:
        bit = STRATEGY_BITS[strategy]
        if bit:
            hit = (mask & bit) != 0
            if hit.any():
                exact[query[hit]] = True

    def build(quotas: dict[str, int]) -> np.ndarray:
        return select_with_quotas(
            query,
            mask,
            n_queries=n_queries,
            cap=spec["cap"],
            quotas=quotas,
            exclude_only=tuple(spec.get("drop_zero_yield", ())),
            cap_mode=spec.get("cap_mode", "rank"),
        )

    return np.where(exact[query], build(spec["with_exact"]), build(spec["quotas"]))


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--store-dir", default=None)
    parser.add_argument("--index-dir", default=None)
    parser.add_argument("--truth", default=None, help="verified-pairs TSV")
    parser.add_argument("--json", default=None, help="write the full report here")
    parser.add_argument("--self-check", action="store_true", default=True,
                        help="verify offline re-slicing matches a real capped run")
    parser.add_argument("--no-self-check", dest="self_check", action="store_false")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    config = ScaleConfig.load(None)
    if args.work_dir:
        config.paths.work_dir = args.work_dir
    config.max_candidates_per_s1 = 0  # 0 == uncapped; selection happens offline

    project = Path(__file__).resolve().parents[1]
    store_dir = Path(args.store_dir) if args.store_dir else config.paths.store_dir(args.split)
    index_dir = Path(args.index_dir) if args.index_dir else config.paths.index_dir(args.split)
    truth_path = Path(args.truth) if args.truth else (
        project / "dataset" / "student_resource" / "dataset" / args.split / f"{args.split}_ground_truth.tsv"
    )
    if not truth_path.exists():
        raise SystemExit(f"ground truth not found: {truth_path}")

    guard = make_guard(config)
    s1 = RecordStore(store_dir, "s1", config)
    pool = PoolStore(store_dir, config)
    indexes = IndexBundle.load(index_dir, mmap=True)
    limit = min(int(args.limit), len(s1))
    LOGGER.info("pool %s | S1 %s | examining first %d rows", f"{pool.n_pool:,}", f"{len(s1):,}", limit)

    frame = s1.read_range(0, limit, ["rowid", "entity_id", "name_norm", "name_core", "addr_core", "country"])
    generator = CandidateGenerator(s1, pool, indexes, config, guard=guard)

    LOGGER.info("one uncapped generation pass (the cap is applied offline after this)")
    began = time.perf_counter()
    batch = generator.generate(0, frame)
    generation_seconds = time.perf_counter() - began
    uncapped_peak = guard.observe()
    if len(batch) == 0:
        raise SystemExit("no candidates generated")
    n_queries = batch.n_queries
    query, pool_rowid, mask = batch.query, batch.pool_rowid, batch.mask
    LOGGER.info(
        "uncapped union: %s candidates (mean %.0f/query, max %d) in %.2fs, peak %s",
        f"{len(batch):,}", batch.per_query.mean(), int(batch.per_query.max()),
        generation_seconds, format_bytes(uncapped_peak),
    )

    ids = frame["entity_id"].to_numpy(dtype=object)
    labels = load_verified_pairs(truth_path, ids)
    truth: dict[int, set[int]] = {}
    for key, matches in labels.items():
        if not matches:
            continue
        slot = np.flatnonzero(ids == key)
        if slot.size == 0:
            continue
        resolved = indexes.pool_ids.resolve(np.array(matches, dtype=object))
        keep = {int(r) for r in resolved.tolist() if r >= 0}
        if keep:
            truth[int(slot[0])] = keep
    LOGGER.info("verified pairs for %d of %d rows", len(truth), limit)

    # -- self-check: offline slicing == a real capped run --------------------
    if args.self_check:
        capped = ScaleConfig.load(None)
        capped.max_candidates_per_s1 = 100
        capped_batch = CandidateGenerator(s1, pool, indexes, capped).generate(0, frame)
        offline = select_with_quotas(query, mask, n_queries=n_queries, cap=100)
        same_rows = capped_batch.pool_rowid.tolist() == pool_rowid[offline].tolist()
        same_queries = capped_batch.query.tolist() == query[offline].tolist()
        if same_rows and same_queries:
            LOGGER.info("self-check OK: offline re-slicing reproduces a real cap=100 run exactly")
        else:
            raise SystemExit("self-check FAILED: offline slicing does not match a real capped run")

    # -- per-strategy attribution over the full union -----------------------
    attribution = _strategy_attribution(query, pool_rowid, mask, truth)
    for row in attribution:
        LOGGER.info(
            "  %-36s %9s candidates  pairs %5s recovered  %5s only-here",
            row["label"], f"{row['candidates']:,}",
            row.get("pairs_recovered", "-"), row.get("pairs_recovered_only_here", "-"),
        )

    # -- where the true matches actually sit in the global rank --------------
    diagnostics = _rank_diagnostics(query, pool_rowid, mask, truth, CAP_SWEEP)
    LOGGER.info("")
    LOGGER.info("WHERE THE TRUE MATCHES SIT IN THE GLOBAL RANK (sole finder per pair)")
    for row in diagnostics:
        recovered = row["recovered_at_cap"]
        LOGGER.info(
            "  %-34s %5d pairs  median rank %6d  p90 %6d  at cap 100: %4d  at cap 200: %4d",
            row["class"], row["verified_pairs"], row["median_global_rank"],
            row["p90_global_rank"], recovered.get("100", 0), recovered.get("200", 0),
        )

    # -- evaluate every configuration ---------------------------------------
    configs = build_configurations() + build_union_configurations()
    configs.append(design_adaptive(attribution, cap=100))
    configs.append(design_adaptive(attribution, cap=100, name="D2_adaptive_union",
                                  cap_mode="union"))
    ceiling = _recall(query, pool_rowid, truth, n_queries)
    LOGGER.info(
        "recall ceiling with no cap at all: q_recall %s pair_recall %s all %s",
        ceiling["query_recall"], ceiling["pair_recall"], ceiling["all_pairs_recall"],
    )
    results = []
    for spec in configs:
        if spec.get("adaptive"):
            t0 = time.perf_counter()
            sel = _select_adaptive(spec, query, mask, n_queries)
            select_seconds = time.perf_counter() - t0
        else:
            t0 = time.perf_counter()
            sel = select_with_quotas(
                query, mask, n_queries=n_queries, cap=spec["cap"],
                quotas=spec["quotas"], exclude_only=tuple(spec.get("exclude_only", ())),
                cap_mode=spec.get("cap_mode", "rank"),
            )
            select_seconds = time.perf_counter() - t0
        kept_query, kept_pool = query[sel], pool_rowid[sel]
        counts = np.bincount(kept_query, minlength=n_queries)
        peak = max(guard.observe(), rss_bytes())
        entry = {
            "name": spec["name"],
            "family": spec["family"],
            "cap": spec["cap"],
            "cap_mode": spec.get("cap_mode", "rank"),
            "quotas": spec["quotas"],
            "exclude_only": list(spec.get("exclude_only", ())),
            "note": spec["note"],
            "candidates": int(kept_pool.size),
            **_percentiles(counts),
            "queries_at_cap": int((counts >= spec["cap"]).sum()),
            "generation_seconds": round(generation_seconds, 2),
            "select_seconds": round(select_seconds, 4),
            "peak_rss_bytes": peak,
            "peak_rss": format_bytes(peak),
            **_recall(kept_query, kept_pool, truth, n_queries),
        }
        results.append(entry)
        LOGGER.info(
            "%-22s cap %3d -> %7s candidates | q_recall %s pair_recall %s all %s",
            spec["name"], spec["cap"], f"{entry['candidates']:,}",
            entry["query_recall"], entry["pair_recall"], entry["all_pairs_recall"],
        )

    frontier = _efficiency_frontier(results)
    report = {
        "split": args.split,
        "s1_rows_examined": limit,
        "pool_rows": pool.n_pool,
        "truth_path": str(truth_path),
        "queries_with_verified_pairs": len(truth),
        "uncapped": {
            "candidates": int(len(batch)),
            **_percentiles(batch.per_query),
            "generation_seconds": round(generation_seconds, 2),
            "peak_rss": format_bytes(uncapped_peak),
            "recall_ceiling": ceiling,
        },
        "strategies": attribution,
        "rank_diagnostics": diagnostics,
        "efficiency_frontier": frontier,
        "configurations": results,
    }
    _print_tables(report)

    if args.json:
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        LOGGER.info("wrote %s", path)
    return 0


def _print_tables(report: dict) -> None:
    uncapped = report["uncapped"]
    print()
    print("=" * 118)
    print(f"candidate experiment -- split={report['split']}  first {report['s1_rows_examined']:,} Source 1 rows")
    print(f"pool {report['pool_rows']:,} rows | {report['queries_with_verified_pairs']:,} rows have verified pairs")
    print("=" * 118)
    print()
    print(f"uncapped union: {uncapped['candidates']:,} candidates "
          f"(mean {uncapped['mean']:.0f}, p50 {uncapped['p50']:.0f}, p99 {uncapped['p99']:.0f}, "
          f"max {uncapped['max']}) in {uncapped['generation_seconds']}s, peak {uncapped['peak_rss']}")
    print()
    print("PER-STRATEGY ATTRIBUTION (over the full uncapped union)")
    print(f"  {'strategy':36s} {'candidates':>12s} {'share':>7s} {'pairs rec.':>11s} {'only here':>10s} {'recall if dropped':>17s}")
    for row in report["strategies"]:
        print(f"  {row['label']:36s} {row['candidates']:>12,} {row['share_of_candidates']:>6.1f}% "
              f"{row.get('pairs_recovered', 0):>11,} {row.get('pairs_recovered_only_here', 0):>10,} "
              f"{row.get('pair_recall_if_removed', 0):>17.4f}")
    print()
    print("CONFIGURATIONS")
    header = (f"  {'config':22s} {'cap':>4s} {'candidates':>11s} {'avg':>7s} {'p50':>5s} {'p95':>5s} "
              f"{'p99':>5s} {'max':>5s} {'q_rec':>7s} {'pair_rec':>9s} {'all_rec':>8s} {'at cap':>7s} {'sel s':>7s} {'peak':>9s}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for row in report["configurations"]:
        print(f"  {row['name']:22s} {row['cap']:>4d} {row['candidates']:>11,} {row['mean']:>7.1f} "
              f"{row['p50']:>5.0f} {row['p95']:>5.0f} {row['p99']:>5.0f} {row['max']:>5d} "
              f"{row['query_recall']:>7.4f} {row['pair_recall']:>9.4f} {row['all_pairs_recall']:>8.4f} "
              f"{row['queries_at_cap']:>7d} {row['select_seconds']:>7.4f} {row['peak_rss']:>9s}")
    ceiling = uncapped.get("recall_ceiling")
    if ceiling:
        print()
        print(f"recall ceiling with no cap at all: pair_recall {ceiling['pair_recall']:.4f} "
              f"| query_recall {ceiling['query_recall']:.4f} | all_pairs {ceiling['all_pairs_recall']:.4f}")
    if report.get("rank_diagnostics"):
        print()
        print("WHERE THE TRUE MATCHES SIT IN THE GLOBAL RANK (sole finder per pair)")
        print(f"  {'sole finder':30s} {'pairs':>7s} {'med rank':>9s} {'p90':>7s} {'@cap100':>8s} {'@cap200':>8s} {'@cap500':>8s}")
        for row in report["rank_diagnostics"]:
            rec = row["recovered_at_cap"]
            print(f"  {row['class']:30s} {row['verified_pairs']:>7,} {row['median_global_rank']:>9,} "
                  f"{row['p90_global_rank']:>7,} {rec.get('100', 0):>8,} {rec.get('200', 0):>8,} {rec.get('500', 0):>8,}")
    if report.get("efficiency_frontier"):
        print()
        print("EFFICIENCY FRONTIER -- is a quota config better than just raising the cap?")
        print(f"  {'config':30s} {'candidates':>11s} {'pair_rec':>9s} {'all_rec':>8s} "
              f"{'cheapest equal plain cap':>25s} {'volume saved':>13s}")
        for row in report["efficiency_frontier"]:
            if row["cheapest_matching_plain_cap"] is None:
                rival = "(beats cap 500)"
                saved = "-"
            else:
                rival = "cap {} ({:,})".format(
                    row["cheapest_matching_plain_cap"], row["cheapest_matching_candidates"]
                )
                saved = "{:.1f}%".format(row["volume_saved_pct"])
            print(f"  {row['config']:30s} {row['candidates']:>11,} {row['pair_recall']:>9.4f} "
                  f"{row['all_pairs_recall']:>8.4f} {rival:>25s} {saved:>13s}")
    print("=" * 118)


if __name__ == "__main__":
    raise SystemExit(main())
