"""Stage 2 check: blocking recall + feature sanity, runnable on any dataset.

    python -m scripts.check_blocking --train-dir dataset/train
"""

from __future__ import annotations

import argparse
import logging
import time
from collections import Counter

import numpy as np
import pandas as pd

from src.blocking import CandidateGenerator, build_pool, strategies_from_mask
from src.config import BlockingConfig, FeatureConfig, setup_logging
from src.data import load_ground_truth, load_split
from src.features import FeatureBuilder, label_from_ground_truth
from src.preprocessing import preprocess_table

LOGGER = logging.getLogger("check_blocking")
RULE = "=" * 96


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", default="dataset/train")
    ap.add_argument("--limit", type=int, default=None, help="limit Source 1 rows for a quick check")
    ap.add_argument("--skip-features", action="store_true")
    args = ap.parse_args()
    setup_logging("INFO")

    train = load_split(args.train_dir, "train")
    s1 = preprocess_table(train.source1)
    s2 = preprocess_table(train.source2)
    s3 = preprocess_table(train.source3)
    if args.limit:
        s1 = s1.head(args.limit).reset_index(drop=True)
    pool = build_pool(s2, s3)
    gt = load_ground_truth(args.train_dir, source1_ids=list(s1["entity_id"]))

    t0 = time.time()
    gen = CandidateGenerator(pool, BlockingConfig())
    cands = gen.generate(s1)
    LOGGER.info("blocking wall time %.1fs", time.time() - t0)

    print(RULE)
    print("CANDIDATE GENERATION STATISTICS")
    print(RULE)
    for k, v in cands.stats.items():
        print(f"  {k:<34} {v}")

    # ---- candidate recall -------------------------------------------------
    print(RULE)
    print("CANDIDATE RECALL")
    print(RULE)
    cands_by_q = cands.candidates_by_query()
    total_true = 0
    found_true = 0
    missing_entities: list[str] = []
    per_entity: list[tuple[str, int, int]] = []
    strategy_hits: Counter[str] = Counter()
    mask_of = {}
    for s1_id, c_ids, mask in zip(cands.query_ids[cands.query_pos], cands.pool_ids[cands.pool_pos], cands.strategy_mask):
        mask_of.setdefault(str(s1_id), []).append((str(c_ids), int(mask)))
    for s1_id in s1["entity_id"].astype(str):
        truth = gt.get(s1_id)
        total_true += len(truth)
        generated = set(cands_by_q.get(s1_id, []))
        found = len(truth & generated)
        found_true += found
        per_entity.append((s1_id, len(truth), found))
        if len(truth) and found < len(truth):
            missing_entities.append(s1_id)
        for cid, mask in mask_of.get(s1_id, []):
            if cid in truth:
                for name in strategies_from_mask(mask):
                    strategy_hits[name] += 1
    recall = found_true / total_true if total_true else 0.0
    print(f"  true pairs                    : {total_true}")
    print(f"  true pairs found in candidates: {found_true}")
    print(f"  CANDIDATE RECALL              : {recall:.4f}")
    perfect = sum(1 for _, t, f in per_entity if t and t == f)
    print(f"  entities with all matches found: {perfect} / {sum(1 for _, t, _ in per_entity if t)}")
    print(f"  entities missing >=1 true match: {len(missing_entities)}")
    print("\n  per-strategy contribution to found true pairs:")
    for name, count in strategy_hits.most_common():
        print(f"    {name:<18} {count:>7}")

    # rank of the first missed true match, to show how close blocking was
    print("\n  examples of blocking failures (true match absent from candidates):")
    shown = 0
    for s1_id in missing_entities:
        truth = gt.get(s1_id)
        generated = set(cands_by_q.get(s1_id, []))
        absent = sorted(truth - generated)
        row = s1[s1["entity_id"].astype(str) == s1_id].iloc[0]
        print(f"    {s1_id} absent={absent} name={str(row.get('business_name'))!r} addr={str(row.get('business_address'))[:60]!r}")
        shown += 1
        if shown >= 10:
            break

    # ---- per-strategy-only recall ---------------------------------------
    print(RULE)
    print("PER-STRATEGY ISOLATED RECALL (what each strategy would achieve alone)")
    print(RULE)
    for name in strategies_from_mask(int(np.bitwise_or.reduce(cands.strategy_mask)) if len(cands.strategy_mask) else 0):
        bit = 1 << ["exact_normalized", "exact_core", "token_name", "tfidf_name", "address_token", "country_aware"].index(name)
        ok = 0
        for s1_id in s1["entity_id"].astype(str):
            for cid, mask in mask_of.get(s1_id, []):
                if (mask & bit) and cid in gt.get(s1_id):
                    ok += 1
        print(f"  {name:<18} {ok:>7} / {total_true}  recall={ok / total_true if total_true else 0:.4f}")

    # ---- features ---------------------------------------------------------
    if not args.skip_features:
        print(RULE)
        print("FEATURE MATRIX")
        print(RULE)
        t0 = time.time()
        fb = FeatureBuilder(s1, pool, FeatureConfig())
        fm = fb.build(cands)
        LOGGER.info("feature build wall time %.1fs", time.time() - t0)
        y = label_from_ground_truth(fm.source1_ids, fm.candidate_ids, gt.matches)
        print(f"  pairs            : {len(fm)}")
        print(f"  features         : {fm.X.shape[1]}")
        print(f"  positives        : {int(y.sum())}  ({y.mean() * 100:.3f}%)")
        print(f"  NaN / inf cells  : {int(np.isnan(fm.X).sum())} / {int(np.isinf(fm.X).sum())}")
        print("\n  class-conditional feature means (positives vs negatives):")
        order = np.argsort(np.abs(fm.X[y == 1].mean(axis=0) - fm.X[y == 0].mean(axis=0)))[::-1]
        for i in order[:18]:
            print(f"    {fm.columns[i]:<38} pos={fm.X[y == 1, i].mean():8.3f}  neg={fm.X[y == 0, i].mean():8.3f}")
        print("\n  constant (zero-variance) features:", [fm.columns[i] for i in range(fm.X.shape[1]) if fm.X[:, i].std() < 1e-9] or "none")


if __name__ == "__main__":
    main()
