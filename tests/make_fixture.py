"""Generate a *synthetic* self-test fixture for the pipeline.

IMPORTANT
---------
This is **not** Amazon challenge data.  It exists only so that the pipeline,
the F0.5 implementation and the blocking/feature code can be executed and
unit-tested without the real files being present.  Nothing here is downloaded,
inferred from the internet, or copied from an external source: every value is
produced by the local generator below from small hand-written word lists.

Once the real ``dataset/`` is available the fixture is never used again.
"""

from __future__ import annotations

import argparse
import random
import re
from pathlib import Path

import pandas as pd

# --- small hand-written vocabularies -----------------------------------------
FIRST_WORDS = [
    "Shree", "New", "Royal", "Ganesh", "Sai", "Krishna", "Modern", "Global", "Sunrise",
    "Crystal", "Silver", "Golden", "Green", "Blue", "Smart", "Prime", "Elite", "Star",
    "Gupta", "Sharma", "Patel", "Reddy", "Nair", "Iyer", "Khan", "Singh", "Das",
    "Apex", "Zenith", "Vertex", "Sigma", "Delta", "Orion", "Nova", "Vertex",
]
SECOND_WORDS = [
    "Traders", "Enterprises", "Solutions", "Services", "Systems", "Technologies",
    "Industries", "Exports", "Imports", "Logistics", "Pharma", "Foods", "Textiles",
    "Motors", "Electronics", "Consultants", "Enterprises", "Agro", "Rice", "Oils",
]
LEGAL_FORMS = [
    "Pvt Ltd", "Private Limited", "pvt ltd.", "Pvt. Ltd.", "Ltd", "Limited",
    "& Sons", "and Sons", "Inc", "Inc.", "Corp", "Corporation", "LLC", "Co", "Company",
]
STREETS = [
    "MG Road", "Nehru Nagar", "Anna Salai", "Park Street", "Gandhi Road", "Station Road",
    "Church Street", "Market Road", "Temple Street", "Lake View Road", "Hill Road",
    "Sector 21", "Plot 44", "Block C", "Tower A", "Near Bus Stand", "Opposite Court",
]
CITIES_IN = [
    ("Bengaluru", "Karnataka", "560001"), ("Mumbai", "Maharashtra", "400001"),
    ("Chennai", "Tamil Nadu", "600001"), ("Hyderabad", "Telangana", "500001"),
    ("Pune", "Maharashtra", "411001"), ("Kochi", "Kerala", "682001"),
    ("Jaipur", "Rajasthan", "302001"), ("Lucknow", "Uttar Pradesh", "226001"),
]
CITIES_US = [
    ("Austin", "TX", "78701"), ("Boston", "MA", "02108"), ("Denver", "CO", "80202"),
    ("Seattle", "WA", "98101"), ("Miami", "FL", "33101"), ("Phoenix", "AZ", "85001"),
    ("Atlanta", "GA", "30301"), ("Portland", "OR", "97201"),
]
# Deliberately confusable pairs: the hardest realistic case for F0.5.
CONFUSABLE_GROUPS = [
    [("Ganesh", "Traders"), ("Ganesh", "Traders"), ("Ganesh", "Trade")],
    [("Royal", "Cafe"), ("Royal", "Cafes"), ("Royal", "Cafe")],
    [("Prime", "Motors"), ("Prime", "Motor"), ("Prime", "Motors")],
    [("Smart", "Tech"), ("Smart", "Technologies"), ("Smart", "Tech")],
    [("Apex", "Foods"), ("Apex", "Food")],
]
NAME_TYPO_POOL = ["s", "x", "a", "ee", "y", "ies", "e", ""]
ADDR_DROP_POOL = [",", ",", ",", " ", " near ", " opposite "]


def _maybe(rng: random.Random, text: str, p: float) -> str:
    return text if rng.random() < p else text


def _typo(rng: random.Random, word: str) -> str:
    if len(word) < 4 or rng.random() > 0.35:
        return word
    pos = rng.randrange(1, len(word) - 1)
    ins = rng.choice(NAME_TYPO_POOL)
    if ins:
        return word[:pos] + ins + word[pos:]
    return word[:pos] + word[pos + 1 :]


def build(n_s1: int, n_pool: int, seed: int, id_prefix: str = "S",
         extra_countries: tuple[str, ...] = ()) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (source1, source2, source3, ground_truth) synthetic frames.

    ``id_prefix`` and ``extra_countries`` exist so the *test* fixture can use a
    disjoint id space and country vocabulary that is unseen in training -- the
    two conditions the real challenge data may exhibit.
    """
    rng = random.Random(seed)
    countries = ("in", "us") + tuple(extra_countries)

    def make_name(core_first: str, core_second: str) -> str:
        legal = rng.choice(LEGAL_FORMS)
        sep = " " if legal.startswith(("&", "and")) else " "
        return f"{core_first} {core_second} {legal}".strip()

    def make_address(country: str) -> str:
        if country in {"in", "ae", "uk"}:
            city, state, pin = rng.choice(CITIES_IN)
            parts = [
                f"{rng.randrange(1, 400)}/{rng.randrange(1, 200)}",
                rng.choice(STREETS),
                _maybe(rng, "Near Bus Stand", 0.3),
                city,
                state,
                pin,
            ]
        else:
            city, state, zcode = rng.choice(CITIES_US)
            parts = [
                f"{rng.randrange(100, 9999)} {rng.choice(STREETS)}",
                city,
                state,
                zcode,
            ]
        addr = ", ".join(parts)
        # strip random components to simulate partial addresses
        for _ in range(rng.randrange(0, 3)):
            if len(addr.split(",")) > 2:
                idx = rng.randrange(len(addr.split(",")))
                segs = addr.split(",")
                segs.pop(idx)
                addr = ",".join(segs)
        return addr

    def canonical_name(first: str, second: str) -> str:
        legal = rng.choice(LEGAL_FORMS)
        return f"{first} {second} {legal}".strip()

    def perturb_name(name: str) -> str:
        """Noisy variant of the *same* business name."""
        tokens = name.split()
        if rng.random() < 0.45:                     # swap the legal form
            tokens = tokens[:-1] + [rng.choice(LEGAL_FORMS)]
        if len(tokens) > 1 and rng.random() < 0.45:  # character noise
            idx = rng.randrange(len(tokens))
            tokens[idx] = _typo(rng, tokens[idx])
        if len(tokens) > 1 and rng.random() < 0.20:  # drop a token
            tokens.pop(rng.randrange(len(tokens)))
        if len(tokens) > 2 and rng.random() < 0.15:  # swap two tokens
            i, j = rng.sample(range(len(tokens)), 2)
            tokens[i], tokens[j] = tokens[j], tokens[i]
        out = " ".join(tokens)
        if rng.random() < 0.20:
            out = out.upper()
        elif rng.random() < 0.20:
            out = out.title()
        return out

    def perturb_address(addr: str) -> str:
        """Noisy variant of the *same* business address."""
        segs = [s.strip() for s in addr.split(",") if s.strip()]
        if len(segs) > 2 and rng.random() < 0.35:     # partial address
            segs.pop(rng.randrange(len(segs)))
        if len(segs) > 2 and rng.random() < 0.20:     # drop the postal code
            segs = [s for s in segs if not s.replace(" ", "").isdigit()]
        if rng.random() < 0.25:                       # street-type abbreviation
            segs = [
                re.sub(r"\b(Road|Road)\b", "Rd", s, flags=re.I) if rng.random() < 0.5 else s
                for s in segs
            ]
        if rng.random() < 0.20:
            segs = [" ".join(_typo(rng, t) for t in s.split()) for s in segs]
        sep = " , " if rng.random() < 0.25 else ", "
        out = sep.join(segs)
        if rng.random() < 0.15:
            out = out.upper()
        return out

    def make_record(entity_id: str, name: str, address: str, country: str, noisy: bool) -> dict:
        """Render a record, optionally as a noisy variant of the given values."""
        if noisy:
            name = perturb_name(name)
            address = perturb_address(address)
            if rng.random() < 0.12:
                name = ""
            if rng.random() < 0.12:
                address = ""
            if rng.random() < 0.06:
                country = ""
        return {
            "entity_id": entity_id,
            "business_name": name,
            "business_address": address,
            "country": country,
        }

    s1_rows: list[dict] = []
    s2_rows: list[dict] = []
    s3_rows: list[dict] = []
    gt_rows: list[dict] = []

    s2_id = s3_id = 0
    for i in range(1, n_s1 + 1):
        s1_id = f"{id_prefix}1-{i:05d}"
        if i % 7 == 0 and CONFUSABLE_GROUPS:
            first, second = rng.choice(CONFUSABLE_GROUPS)[0]
        else:
            first, second = rng.choice(FIRST_WORDS), rng.choice(SECOND_WORDS)
        country = rng.choice(countries)
        name = canonical_name(first, second)
        address = make_address(country)
        s1_rows.append(make_record(s1_id, name, address, country, noisy=False))
        matches: list[str] = []
        # cardinality distribution: ~35% singletons, ~40% one match, ~25% many
        bucket = rng.random()
        n_matches = 0 if bucket < 0.35 else (1 if bucket < 0.75 else rng.randint(2, 4))
        for _ in range(n_matches):
            if rng.random() < 0.6:
                s2_id += 1
                eid = f"{id_prefix}2-{s2_id:05d}"
                s2_rows.append(make_record(eid, name, address, country, noisy=True))
            else:
                s3_id += 1
                eid = f"{id_prefix}3-{s3_id:05d}"
                s3_rows.append(make_record(eid, name, address, country, noisy=True))
            matches.append(eid)
        gt_rows.append({"source1_id": s1_id, "matched_entity_ids": ",".join(matches)})

        # Hard negative: a *different* business with the same name core in the
        # same country.  This is the classic precision trap, and it is what an
        # address-aware model has to learn to reject.
        if rng.random() < 0.18:
            other_address = make_address(country)
            while other_address.split(",")[-1].strip() == address.split(",")[-1].strip():
                other_address = make_address(country)
            s2_id += 1
            s2_rows.append(
                make_record(f"{id_prefix}2-{s2_id:05d}", name, other_address, country, noisy=True)
            )

    # Distractor pool records that match nothing in source 1.
    for _ in range(n_pool):
        name = canonical_name(rng.choice(FIRST_WORDS), rng.choice(SECOND_WORDS))
        address = make_address(rng.choice(countries))
        if rng.random() < 0.5:
            s2_id += 1
            s2_rows.append(make_record(f"{id_prefix}2-{s2_id:05d}", name, address, rng.choice(countries), noisy=True))
        else:
            s3_id += 1
            s3_rows.append(make_record(f"{id_prefix}3-{s3_id:05d}", name, address, rng.choice(countries), noisy=True))
    return (
        pd.DataFrame(s1_rows),
        pd.DataFrame(s2_rows),
        pd.DataFrame(s3_rows),
        pd.DataFrame(gt_rows),
    )


def build_test(n_s1: int, n_pool: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Test split: same record generator, no ground truth."""
    s1, s2, s3, _ = build(n_s1, n_pool, seed, id_prefix="T", extra_countries=("ae", "uk"))
    return s1, s2, s3


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate a synthetic self-test fixture (NOT challenge data).")
    ap.add_argument("--out", default="dataset", help="dataset root to write into")
    ap.add_argument("--train-s1", type=int, default=1500)
    ap.add_argument("--train-pool", type=int, default=2500)
    ap.add_argument("--test-s1", type=int, default=600)
    ap.add_argument("--test-pool", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20240101)
    args = ap.parse_args()

    root = Path(args.out)
    (root / "train").mkdir(parents=True, exist_ok=True)
    (root / "test").mkdir(parents=True, exist_ok=True)

    s1, s2, s3, gt = build(args.train_s1, args.train_pool, args.seed)
    for frame, name in ((s1, "train_source1.tsv"), (s2, "train_source2.tsv"),
                        (s3, "train_source3.tsv"), (gt, "train_ground_truth.tsv")):
        frame.to_csv(root / "train" / name, sep="\t", index=False)
    t1, t2, t3 = build_test(args.test_s1, args.test_pool, args.seed + 1)
    for frame, name in ((t1, "test_source1.tsv"), (t2, "test_source2.tsv"), (t3, "test_source3.tsv")):
        frame.to_csv(root / "test" / name, sep="\t", index=False)
    print(f"fixture written to {root}: train s1={len(s1)} s2={len(s2)} s3={len(s3)} gt={len(gt)}")


if __name__ == "__main__":
    main()
