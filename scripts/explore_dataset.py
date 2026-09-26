"""Stage 1 - dataset exploration.

Prints shape / columns / dtypes / missing-value statistics / duplicate
statistics / representative records / ground-truth cardinality distribution /
country distribution, for both splits.  Works on the real challenge files or
on any TSV set with the same schema.

Usage
-----
    python -m scripts.explore_dataset
    python -m scripts.explore_dataset --train-dir dataset/train --test-dir dataset/test
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from src.config import ADDRESS_COLUMN, COUNTRY_COLUMN, ID_COLUMN, NAME_COLUMN, setup_logging
from src.data import load_ground_truth, load_split
from src.data.loader import describe_dataframe
from src.data.validator import log_warnings, validate_dataset
from src.preprocessing import normalize_address, normalize_country, normalize_name

LOGGER = logging.getLogger("explore")

RULE = "=" * 100


def _s(value: object) -> str:
    """Safe string conversion that tolerates ``None``/``pd.NA``/NaN."""
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)


def _section(title: str) -> None:
    print(f"\n{RULE}\n{title}\n{RULE}")


def _frame_report(df: pd.DataFrame, label: str, show: int = 5) -> None:
    stats = describe_dataframe(df, label)
    print(f"\n--- {label} ---")
    print(f"shape                 : {stats['shape']}")
    print(f"columns               : {stats['columns']}")
    print("dtypes                :")
    for c, t in stats["dtypes"].items():  # type: ignore[union-attr]
        print(f"    {c:<28} {t}")
    print("missing values        :")
    nulls = stats["null_counts"]  # type: ignore[assignment]
    fracs = stats["null_fraction"]  # type: ignore[assignment]
    for c in nulls:  # type: ignore[union-attr]
        print(f"    {c:<28} nulls={nulls[c]:<8} ({fracs[c] * 100:5.2f}%)")
    if "id_unique" in stats:
        print(f"unique ids            : {stats['id_unique']}  duplicated id rows={stats['id_duplicates']}")
    for c in (NAME_COLUMN, ADDRESS_COLUMN, COUNTRY_COLUMN):
        if f"{c}_unique" in stats:
            print(f"{c:<22}: unique={stats[f'{c}_unique']:<8} empty_string={stats[f'{c}_empty_string']:<7} "
                  f"len_mean={stats[f'{c}_len_mean']} len_median={stats[f'{c}_len_median']}")
    print(f"representative records (first {show}):")
    with pd.option_context("display.max_columns", None, "display.width", 200, "display.max_colwidth", 60):
        print(df.head(show).to_string(index=False))


def _normalized_examples(df: pd.DataFrame, label: str, n: int = 8) -> None:
    print(f"\n--- {label}: raw -> normalized examples ---")
    sub = df.head(n)
    for _, row in sub.iterrows():
        raw_name = _s(row.get(NAME_COLUMN, ""))
        raw_addr = _s(row.get(ADDRESS_COLUMN, ""))
        raw_cty = _s(row.get(COUNTRY_COLUMN, ""))
        print(f"  name    : {raw_name!r}\n            -> {normalize_name(raw_name)!r} (core)")
        print(f"  address : {raw_addr[:80]!r}\n            -> {normalize_address(raw_addr)[:110]!r}")
        print(f"  country : {raw_cty!r} -> {normalize_country(raw_cty)!r}")
        print()


def _ground_truth_report(gt, s1_ids: list[str]) -> dict:
    _section("GROUND TRUTH ANALYSIS")
    hist = gt.counts_by_cardinality()
    n = len(gt)
    print(f"annotated Source 1 entities : {n}")
    total_pairs = sum(len(v) for v in gt.matches.values())
    print(f"total true pairs            : {total_pairs}")
    print(f"mean matches per entity     : {total_pairs / n if n else 0:.3f}")
    print("\nmatches-per-entity histogram:")
    for k, v in hist.items():
        bucket = {"0": "zero matches (must stay unmatched)",
                  "1": "exactly one match",
                  "2": "two matches"}.get(k, f"{k} matches")
        print(f"    {k:>3} matches : {v:>7} entities  ({v / n * 100 if n else 0:5.2f}%)   {bucket}")
    ent = gt.entity_summary()
    print("\nmatch-count quantiles:", {q: float(ent['n_matches'].quantile(q)) for q in (0.25, 0.5, 0.75, 0.9, 0.99)})
    print("\nexamples:")
    for s1 in gt.source1_ids[:5]:
        print(f"    {s1:<12} -> {sorted(gt.get(s1))[:6]}")
    return {"n_entities": n, "n_pairs": total_pairs, "cardinality_histogram": hist}


def _country_report(tables: dict[str, pd.DataFrame]) -> dict:
    _section("COUNTRY DISTRIBUTION (open-set, derived from data)")
    out: dict[str, dict[str, int]] = {}
    for label, df in tables.items():
        if COUNTRY_COLUMN not in df.columns:
            continue
        raw = df[COUNTRY_COLUMN].fillna("").astype(str)
        norm = raw.map(normalize_country)
        counts = norm.value_counts()
        out[label] = {str(k): int(v) for k, v in counts.items()}
        print(f"\n{label}: {counts.size} distinct normalized values")
        print(counts.head(25).to_string())
        print(f"    missing/empty: {int((norm == '').sum())} ({(norm == '').mean() * 100:.2f}%)")
    train_vals = set(out.get("source1", {})) | set(out.get("source2", {})) | set(out.get("source3", {}))
    test_vals = set(out.get("source1_test", {}))
    if test_vals - train_vals:
        print(f"\n!! open-set countries in test not seen in train: {sorted(test_vals - train_vals)}")
    else:
        print("\nno test country value is unseen in train (this is data-dependent, not assumed)")
    return out


def _duplicate_report(tables: dict[str, pd.DataFrame]) -> None:
    _section("DUPLICATE STATISTICS")
    for label, df in tables.items():
        if ID_COLUMN not in df.columns:
            continue
        print(f"\n{label}:")
        print(f"    fully duplicated rows            : {int(df.duplicated(keep=False).sum())}")
        print(f"    duplicated ids                   : {int(df[ID_COLUMN].duplicated(keep=False).sum())}")
        if NAME_COLUMN in df.columns:
            print(f"    duplicated (id, name) pairs      : {int(df.duplicated(subset=[ID_COLUMN, NAME_COLUMN], keep=False).sum())}")
        nn = df[NAME_COLUMN].fillna("").astype(str) if NAME_COLUMN in df.columns else None
        if nn is not None and len(nn):
            vc = nn.value_counts()
            print(f"    most repeated raw name          : {vc.index[0]!r} x{vc.iloc[0]}" if vc.iloc[0] > 1 else "    no repeated raw name")
            nn_norm = nn.map(lambda v: normalize_name(v))
            vcn = nn_norm.value_counts()
            if vcn.iloc[0] > 1:
                print(f"    most repeated normalized name   : {vcn.index[0]!r} x{vcn.iloc[0]}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Explore the entity-resolution dataset.")
    ap.add_argument("--train-dir", default="dataset/train")
    ap.add_argument("--test-dir", default="dataset/test")
    ap.add_argument("--json-out", default=None, help="optional path for a machine-readable summary")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    setup_logging(args.log_level)

    summary: dict = {"train": {}, "test": {}}

    _section("TRAINING SPLIT")
    train = load_split(args.train_dir, "train")
    train_tables = dict(train)
    log_warnings(validate_dataset(train_tables))
    for label, df in train_tables.items():
        _frame_report(df, f"train {label}")
        summary["train"][label] = describe_dataframe(df, label)

    _section("TEST SPLIT")
    test = load_split(args.test_dir, "test")
    test_tables = dict(test)
    log_warnings(validate_dataset(test_tables))
    for label, df in test_tables.items():
        _frame_report(df, f"test {label}")
        summary["test"][label] = describe_dataframe(df, label)

    _section("NORMALIZATION SPOT-CHECK")
    _normalized_examples(train.source1, "train source1")
    _normalized_examples(train.source2, "train source2", n=3)
    _normalized_examples(train.source3, "train source3", n=3)

    gt = load_ground_truth(args.train_dir, source1_ids=list(train.source1[ID_COLUMN]))
    summary["ground_truth"] = _ground_truth_report(gt, list(train.source1[ID_COLUMN]))

    _duplicate_report(train_tables)
    _duplicate_report(test_tables)
    summary["countries"] = _country_report({**train_tables, **{f"{k}_test": v for k, v in test_tables.items()}})

    # leakage sanity check: are train and test ids disjoint?
    tr_ids = set(train.source1[ID_COLUMN]) | set(train.source2[ID_COLUMN]) | set(train.source3[ID_COLUMN])
    te_ids = set(test.source1[ID_COLUMN]) | set(test.source2[ID_COLUMN]) | set(test.source3[ID_COLUMN])
    _section("TRAIN / TEST OVERLAP")
    print(f"train ids: {len(tr_ids)}   test ids: {len(te_ids)}   shared: {len(tr_ids & te_ids)}")
    summary["overlap"] = {"n_train_ids": len(tr_ids), "n_test_ids": len(te_ids), "shared": len(tr_ids & te_ids)}

    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        print(f"\nJSON summary written to {args.json_out}")


if __name__ == "__main__":
    main()
