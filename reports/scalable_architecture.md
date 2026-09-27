# Scalable Architecture — Amazon Business Entity Resolution

Status: design approved; data/index/candidate layer implemented and measured on
the real train split (see "Implemented in this stage" for the numbers). Feature
generation, training, inference, and output writing are deferred.
Scope of this document: the whole pipeline (A–K), so the deferred stages are
specified before they are written.

---

## A. Dataset scale

| table | rows | file size |
| --- | ---: | ---: |
| `train_source1` | 2,206,821 | 200 MB |
| `train_source2` | 5,034,616 | 467 MB |
| `train_source3` | 5,285,603 | 480 MB |
| `train_ground_truth` (wide) | 2,206,821 | 121 MB |
| `test_source1` | 1,732,544 | 167 MB |
| `test_source2` | 4,887,273 | 486 MB |
| `test_source3` | 5,082,316 | 483 MB |

Derived quantities:

| quantity | train | test |
| --- | ---: | ---: |
| retrieval pool `N = S2 ∪ S3` | **10,320,219** | **9,969,589** |
| queries `|Q|` (= Source 1 rows) | 2,206,821 | 1,732,544 |
| brute-force pair space `|Q| × N` | **2.28 × 10¹³** | **1.73 × 10¹³** |
| verified positive pairs | 7,638,365 | — |
| true singletons (no match) | 123,247 (5.59%) | — |

The ground truth is **wide** (`source1_entity_id`, `matched_entity_ids` with a
comma-separated list) and **many-to-many**: the mean entity carries
7,638,365 / (2,206,821 − 123,247) ≈ **3.67** distinct Source 2/3 records. A
Source 1 entity may legitimately receive several matches, so the decision rule
is not top-1.

The single number that governs every design decision below is
`|Q| × N ≈ 2.3 × 10¹³`. Any algorithm that touches every query against every
pool record is dead on arrival. Everything in this design is **linear in the
number of records**, with a per-query work budget instead of a per-corpus scan.

## B. Memory constraints

Host: 15.6 GB RAM, 20 logical CPUs, Python 3.10, `pyarrow` 23 (Parquet),
`polars`, `psutil`, stdlib `sqlite3`. No DuckDB.

Raw text alone is 2.4 GB. Python object-dtype strings cost roughly
`49 + len` bytes each plus an 8-byte pointer, so the *same* text held as
`pandas` object columns costs an estimated **4–6 GB**. Any architecture that
also holds a candidate-pair table, a feature matrix, and per-pair Python
dictionaries cannot fit.

Working limits adopted for every stage:

| limit | value | rationale |
| --- | ---: | --- |
| peak RSS budget | **6.0 GB** | ~38% of physical RAM, leaves headroom for Parquet write buffers and the OS page cache |
| TSV read chunk | 500,000 rows | ~250 MB of text, pandas parser needs ~3× row width |
| store write row-group | 250,000 rows | keeps Parquet shards ~15–25 MB, independently readable |
| Source 1 query batch | 10,000 rows | × per-query posting budget → bounded candidate array |
| per-query posting budget | 3,000 name / 1,500 address | the *only* thing that bounds candidate generation |
| candidates kept per query | 100 (configurable) | ≈ 220 M pairs on train, 2.9 GB on disk, never in RAM |
| training pairs (sampled) | ≤ 12 M | 12 M × 62 × 4 B = 3.0 GB feature matrix |
| in-memory Python objects from data | 0 | all per-row state is a numpy column, never a list/dict of rows |

Rules enforced in code:

* original TSVs are opened **read-only** and never rewritten;
* nothing larger than one chunk is a `pandas.DataFrame`;
* no `pickle`/`joblib`/`np.save` of a per-record object — the only serialised
  artefacts are Parquet shards and flat `.npy` integer/float arrays;
* every stage is resumable from a manifest and reports rows, elapsed seconds,
  candidate counts and RSS on a fixed interval.

## C. Blocking strategy

The previous design ran five strategies, three of which were
`NearestNeighbors(algorithm="brute")` over the *entire* pool — an implicit
`|Q| × N` sparse dot product, repeated. Replaced by **selective inverted
indexes**: an index is built once from the pool, and a query only ever reads
the postings of its own rarest tokens.

Six generators, each with a hard per-query budget:

| id | generator | index | per-query budget | score |
| --- | --- | --- | --- | --- |
| `E1` | exact `name_normalized` | sorted key-hash table | 1,000 block size | 1.0 |
| `E2` | exact `name_core` | sorted key-hash table | 1,000 block size | 1.0 |
| `T1` | rarest name tokens | inverted index, df-capped | 3,000 postings | IDF cosine |
| `T2` | rarest address tokens | inverted index, df-capped | 1,500 postings | IDF cosine |
| `SN` | char-3gram / sorted neighbourhood over `name_core` | lexicographically sorted pool names | 50 window | rapidfuzz ratio |
| `C1` | same-country filter over `T1` postings | country partition | reuses T1 budget | IDF cosine + country bonus |

Why these and not character TF-IDF:

* `char_wb` 2–5-gram TF-IDF over 10.3 M pool names is ≈ 620 M non-zeros
  (`float32` + `int32`) ≈ **7.5 GB** per field, and the address field doubles
  it. It cannot be fitted.
* Its query cost is a brute-force sparse product over the whole corpus, which
  is the O(N²) we are removing.
* **Sorted neighbourhood** gives most of the recall at O(N log N) build and
  O(log N + W) per query, using a lexicographic sort instead of a vocabulary.
  It is the standard answer to "find near-duplicate strings at scale", and it
  degrades gracefully: a transposed or truncated name stays adjacent in sort
  order.
* rapidfuzz is applied **only** to the ≤ 50 window returned by `SN`, i.e. only
  to already-blocked candidates. It is never run over the full pool.

Token selection is the "rarest first" rule, and it is what makes the budget
work: a query contributes at most `max_tokens_per_query` (3) postings lists,
each capped at `max_df_index` (default 5,000 pool rows), and expansion stops as
soon as the accumulated budget is spent. Generic names ("services", "trading")
therefore cost nothing: their df is either below `min_df` or their postings
list is never reached because a rarer token is expanded first.

## D. Index design

All indexes live under `work/<split>/index/` as flat, memory-mappable arrays.
There is no pickle and no object array.

### D.1 Inverted token index (one per field: name, address)

Postings are packed into a **single sorted `int64` array**:

```
posting = (term_id << 32) | pool_rowid        # pool_rowid < 2^32
```

Sorting by the composite gives, for free, both the grouping by `term_id` and
ascending `pool_rowid` inside each group — which is what the per-query
`searchsorted` and the country intersection need.

| array | dtype | size (name, 10.3 M rows) | purpose |
| --- | --- | ---: | --- |
| `postings.npy` | `int64` | ~330 MB (≈ 4.1 postings/row) | the postings themselves |
| `offsets.npy` | `int64` | `8 × (T+1)` | start of each term's posting run |
| `df.npy` | `int32` | `4 × T` | document frequency, used to rank terms |
| `idf.npy` | `float32` | `4 × T` | `log((1+N)/(1+df)) + 1`, fitted on the pool only |
| `doc_norm.npy` | `float32` | `4 × N` | L2 norm of each pool row, precomputed once |
| `doc_terms.npy` | `int8` | `1 × N` | in-index term count per row |
| `terms.parquet` | string | ~40 MB | `term_id → term`, for diagnostics and explainability |

`T` (distinct name terms) is expected in the 1–3 M range, so the per-term
side-tables are tens of MB. Everything is written with
`np.save`/`np.load(mmap_mode="r")`, so the OS page cache — not the heap —
absorbs the 330 MB.

Two IDF-weighted quantities are needed per query and both are precomputed
rather than recomputed:

* **query side** — norm from the query's own terms, O(terms).
* **pool side** — `doc_norm[rowid]`, O(1) lookup.

This is why the per-query cosine costs one `searchsorted` + one `bincount`
and never a document scan.

### D.2 Exact-key index (`name_normalized`, `name_core`)

Same packing, one posting per row:

```
key = (blake2b_64(key_string) << 32) | pool_rowid
```

Keys shorter than 2 characters are skipped (a block of `""` is a
degenerate mega-block). 64-bit hashing makes a dictionary of 10.3 M Python
strings unnecessary — the table is a sorted `int64` array plus offsets, ~82 MB
per field. Hash collisions at 2⁻⁶⁴ are not a correctness concern for a
*recall-oriented* index: a collision can only add a candidate, and the
similarity model filters it.

### D.3 Sorted-neighbourhood index

```
sort_key  = name_core padded/truncated to `sn_key_chars` (default 12)
sn_keys   = fixed-width byte array, lexicographically sorted   (≈ 130 MB)
sn_rows   = int32[N], pool_rowid in that same order             (≈ 41 MB)
```

A query computes its own 12-byte key, `searchsorted`s for the insertion
point, and expands left and right while the running common prefix is
≥ `sn_prefix_chars` (6) or the window `sn_window` (50) is exhausted. Cost per
query: one binary search plus ≤ 100 string comparisons. This is the only
"character/ngram retrieval" and it is bounded by construction.

### D.4 Country index

`country → sorted pool_rowid`, one `(ids, offsets)` pair (~41 MB total, ~200
groups). Used as a **filter**, not as a generator: the ≤ 3,000 postings already
gathered for `T1` are intersected with the query's country group via
`searchsorted` (O(k log N), no scan of the group). It also supplies the
country match/conflict features and the open-set diagnostics.

### D.5 Id → rowid resolver

Needed for ground-truth labelling and for output. A Python `dict` of 12.5 M
strings is ~1.8 GB, so ids are stored as a **sorted fixed-width byte array**
plus a parallel `int32` rowid array, and resolved with `np.searchsorted`:

```
id_keys  = sorted fixed-width ids      (12.5 M × L bytes, L from the data)
id_rows  = int32 rowids, aligned       (50 MB)
```

Peak ≈ 210 MB for L = 13, and resolution of 7.6 M targets is one vectorised
`searchsorted`.

### D.6 Record store

`work/<split>/store/{s1,s2,s3}/part-*.parquet`, one shard per 500 k input
rows, schema:

```
rowid      int32   global row index within the split (pool: s2 then s3)
entity_id  string
name_norm  string   business_name_normalized
name_core  string   business_name_core
addr_norm  string
addr_core  string
country    string   country_normalized
pincode, house_number, city, state  string
```

`rowid` is the join key everywhere downstream: the candidate tables,
ground-truth pairs and output files are all integer arrays indexed by
`rowid`. Entity ids are only decoded to text at the output boundary. Total on
disk ≈ 600 MB per split — cheap, and read back column-by-column via
`pyarrow` so a feature batch touches only the columns it needs.

## E. Candidate generation

For each batch of ≤ 10,000 Source 1 rows:

1. **Expand** each generator into flat `int32` arrays
   `(q_local, pool_rowid, score_float32, strategy_bit_uint8)`. No dicts, no
   lists of tuples — the previous `best`/`mask`/`origin` triple-dict union is
   gone.
2. **Deduplicate** by `pair_key = q_local * N + pool_rowid` with
   `np.unique(..., return_inverse=True)`, then
   `np.maximum.at` for the score and `np.bitwise_or.at` for the strategy mask.
   Cost O(M log M) in the batch's M pairs, memory O(M).
3. **Cap** to `max_candidates_per_s1` by sorting on
   `lexsort((pool_rowid, -score, q_local))` and slicing at group boundaries.
   The strongest candidates always survive.
4. **Emit** a Parquet shard `candidates/part-*.parquet` with
   `(s1_rowid int32, pool_rowid int32, score float32, mask uint8)` — 13 bytes
   per pair, compressed.

Nothing is accumulated across batches. A 2.2 M-query run produces ~220 M
candidate rows **on disk** (~2.9 GB) in shards of ~10 M rows, and peak memory
is one batch.

For training, positives are injected explicitly: any verified true pair that
the blocking missed is appended to the shard, so candidate recall for the
training set is 1.0 by construction and the classifier sees every positive.
Recall is still *measured* (not assumed) for the test split, where no labels
exist.

## F. Feature generation

Feature computation is a pure function of the candidate shard plus two
`searchsorted` gathers from the store, so it is embarrassingly parallel over
shards and bounded by the shard size.

* Field values are fetched as numpy string arrays, never as row objects. The
  previous `_record_lookup` (`frame.iterrows()` over 10.3 M rows → 10.3 M
  `pd.Series`) and the 10.3 M-dict `_components` list are not reproduced.
* rapidfuzz runs via `process.cdist` **per query group, over that group's
  candidates only** — 10² comparisons, not 10⁷.
* The string-preparation cache is keyed on the *distinct* normalized values of
  the shard and is dropped at shard end, so it cannot grow to 62 M
  `StringFeatures` objects as the old `name_builder.cache()` calls did.
* The TF-IDF cosine features use the same `doc_norm` array as blocking, but
  restricted to the shard's candidate rows.
* Output: `features/part-*.parquet`, `(s1_rowid, pool_rowid, f0..f61 float32)`.
  Never a dense `N × N` anything, never a full-run `np.zeros((n_pairs, 62))`.

**Not implemented in this stage** — the design is: ~62 columns, all defined
even when a field is missing so the matrix is NaN-free, no hand-tuned weights
(combined features are parameter-free summaries), the same feature layout at
train and inference time, guarded by a schema hash.

## G. Training-pair construction

Featurising all ~220 M candidate pairs is 220 M × 62 × 4 B ≈ 55 GB of Parquet.
It is also unnecessary: 7.6 M of those pairs are positive and the rest are
overwhelmingly trivial negatives.

Sampling plan, executed shard-by-shard with a fixed seed:

1. **All** candidate pairs that are verified positives → 7.6 M (never sampled
   away; the old code's "keep everything" rule, now affordable by sampling the
   *negatives* instead).
2. **Hard negatives**: the top `hard_negatives_per_query` (default 5) non-
   positive candidates per query by blocking score → the confusable
   near-misses that carry the discriminative signal.
3. **Random negatives**: a reservoir sample of the remainder, scaled to reach
   `max_train_pairs` (default 12 M) total, stratified so every source-1
   cardinality bucket (0, 1, 2, 3+ true matches) is represented. This is what
   teaches the model to reject, which is where F0.5 is won: 123,247 singletons
   × a false positive each is a hard zero.
4. **All** pairs of singleton entities at negative rate 1 up to a cap, since
   they are the metric's dominant risk and are otherwise a 5.6% minority.

Labels come from the ground-truth pair array, joined on `(s1_rowid,
pool_rowid)` — an integer join, not a string-set membership test per pair.
A verified positive can never be labelled 0.

The resulting design matrix is 12 M × 62 `float32` = 3.0 GB and fits in RAM;
it is written as Parquet shards first so the sample is reproducible and
inspectable before anything is fitted.

## H. Model training

K-fold cross-validation **on Source 1 entities** (the macro-averaging unit),
never on pairs, so a pair cannot straddle a fold. The retrieval pool and its
IDF statistics are deliberately shared across folds: they are label-free
functions of the record data and are exactly what is available at inference.

Each fold trains on its own 75% row subset of the sampled matrix. Because the
matrix is 3.0 GB, `HistGradientBoostingClassifier` is fitted on
`float32` numpy directly and models are compared on **out-of-fold F0.5** with
the threshold swept over the exact observed score breakpoints (a complete
threshold set — a coarse grid can miss the optimum).

Class imbalance: `class_weight="balanced"` plus the sampling in §G, which
controls the positive rate explicitly instead of leaving it to a 100:1 accident.

The winner is refit on 100% of the sampled matrix. Only the fitted estimator,
the feature schema hash and the threshold are serialised — a few hundred kB.

## I. Inference

Identical machinery on the test split, with the test pool's own indexes (its
IDF must come from the test pool, exactly as train IDF came from the train
pool):

```
build_store(test) → build_index(test) → per-batch candidates
    → per-shard features (streaming) → model.predict_proba per shard
    → threshold per shard → accumulate only the surviving matches
```

Peak memory is one shard. The only cross-shard state is the set of accepted
`(pool_rowid)` matches for the current Source 1 batch, which is bounded by
`batch × cap`.

## J. Output generation

Both files are **streamed**, not built as a DataFrame:

* `matching_results.tsv` — one line per test Source 1 row, in file order,
  written as each batch completes. 1.73 M lines, ~40 MB. The old
  `build_matching_results_frame` built a 1.7 M-element list of dicts first.
* `candidate_pairs.tsv` — same streaming path; ~2 GB, so it is written
  optionally (`--emit-candidate-pairs`) but the format and the
  "every emitted match appears here" invariant are preserved.

The validator runs on the written files in streaming mode too; a full-file
`pd.read_csv` of a 2 GB TSV is itself a memory bug.

Entity ids are decoded from `rowid` through the id resolver only at this
boundary.

## K. Expected computational complexity

`Q` = Source 1 rows, `N` = pool rows, `T` = distinct terms, `P` = postings,
`t` = terms per record, `W` = SN window, `b` = per-query posting budget,
`C` = candidates kept per query, `M` = sampled training pairs.

| stage | build | query | notes |
| --- | --- | --- | --- |
| normalise + store | `O(N)` | — | streamed, one 500 k chunk resident |
| inverted index | `O(P log P)`, `P = O(N·t)` | `O(b + b log N)` | `searchsorted` + `bincount` |
| exact-key index | `O(N log N)` | `O(log N + k)` | |
| sorted neighbourhood | `O(N log N)` | `O(log N + W)` | 12-byte fixed-width keys |
| country index | `O(N)` | `O(b log N)` | filter only |
| id resolver | `O(N log N)` | `O(M log N)` | |
| **candidate generation** | — | **`O(Q · b)`** | `b ≤ 3,000`, not `O(Q·N)` |
| dedup + cap | — | `O(Q·b log(Q·b))` per batch | vectorised, no dicts |
| features | — | `O(C · Q)` | rapidfuzz on blocked pairs only |
| training | `O(M)` | — | 1 split, ≤ 12 M rows |
| inference | `O(Q_test · b)` | — | streamed |

**Headline:** retrieval goes from `O(Q · N) = 2.3 × 10¹³` to
`O(Q · b) ≈ 6.6 × 10⁹` posting reads for the whole train split — a factor of
~3,400 — and the constant is a *configurable budget* rather than the corpus
size. Peak memory is `O(chunk + b · batch)` and is independent of `Q` and `N`.

---

## Why the previous approach could not work

| # | location | problem | cost at this scale |
| --- | --- | --- | --- |
| 1 | `src/data/loader.py:319` `load_split` | all three tables fully materialised as object-dtype strings | 4–6 GB for train, ×2 with test |
| 2 | `src/pipeline/train_pipeline.py:175` `_record_lookup` | `frame.iterrows()` over 10.3 M rows | 10.3 M `pd.Series` ≈ 10–20 GB |
| 3 | `src/pipeline/train_pipeline.py:161-162` | `pd.concat` of S2+S3 again, just for that lookup | +2 GB peak |
| 4 | `src/pipeline/stages.py:57-60` + `preprocess.py:47` | `df.copy()` per table, then 9 more object columns each | 2× raw + normalised |
| 5 | `src/blocking/tfidf_blocking.py:58-80` | `NearestNeighbors(algorithm="brute")` on the whole pool | O(\|Q\|·N) = 2.3e13 |
| 6 | same, called from `candidate_generator.py:195, 211, 239, 254` | four more full-corpus sparse products | 4 × O(\|Q\|·N) |
| 7 | `src/blocking/character_blocking.py:41` | `char_wb` 2–5-gram TF-IDF over 10.3 M rows, twice | ≈ 7.5 GB per field → OOM |
| 8 | `src/features/feature_builder.py:174-181` | four more full-corpus vectorizer fits on a `concatenate`d 12.5 M-row list | OOM + hours |
| 9 | `src/blocking/token_blocking.py:108-112` | per-country matrix slice **and** a fresh `SparseRetriever` fit, once per country | 200 re-fits, O(countries × Q) scans |
| 10 | `src/features/feature_builder.py:212` | `X = np.zeros((n_pairs, 62))` for the whole run (≈ 883 M × 62) | 219 GB |
| 11 | `src/blocking/candidate_generator.py:296-306` | `best`/`mask`/`origin` Python dicts over all pairs (`origin` never even read) | 100+ GB |
| 12 | `src/features/feature_builder.py:153-158` | `to_dict("records")` → a dict per pool row | 10.3 M dicts ≈ 4 GB |
| 13 | `src/features/feature_builder.py:226-264` | per-entity Python loop, `np.split` per group, `.iloc[q]` per entity | 2.2 M iterations, 6 cdist calls each |
| 14 | `src/models/predict.py:50-55` | `score_map: dict[str, dict[str, float]]` holding every pair's score | 883 M nested entries |
| 15 | `src/evaluation/validation.py:131-137` | `generated` + `masks` nested dicts over all pairs | 2 × 883 M entries |
| 16 | `src/evaluation/validation.py:189` | 10.3 M-entry dict of ids→source just for a lookup | 1.5 GB |
| 17 | `src/data/ground_truth.py:84-86` | `positive_pairs()` → a 7.6 M-tuple Python set | 1.5–2 GB |
| 18 | `src/pipeline/train_pipeline.py:163` | `pd.DataFrame(X)` copy of the whole feature matrix, for error analysis | +219 GB |
| 19 | `src/pipeline/train_pipeline.py:232-237` | full pair-level score dump then `sort_values` | ~30 GB TSV |
| 20 | `src/blocking/candidate_generator.py:185-186, 203-208` | the 10.3 M-row pool is re-tokenised on every call | 2× the work, 2× the memory |
| 21 | `reports/dataset_profile_real.json` | already produced a 390 MB report | symptom of the same pattern |
| 22 | `scripts/explore_dataset.py:163-173, 191-192` | train *and* test both fully resident, then 2 × 12 M-entry id sets | >10 GB |
| 23 | `notebooks/01_data_exploration.ipynb` cell 17 | re-reads all of S2+S3 id columns after `profile_tsv` already streamed them | 2 extra full passes |

The redesigned layer removes 1–23 by construction: nothing is ever
materialised whole, every join is on `int32` row ids, every set operation is a
sorted `int64` array, and every batch is bounded by a configuration value.

---

## Implemented in this stage

Only the data/index layer (sections B, C, D, E) plus the label resolver from §G.
Feature generation, training-pair sampling, model training, inference, output
writing, and the Streamlit app are **not** implemented.

```
src/scale/
  config.py       ScaleConfig — every budget, defaults justified in the docstring
  progress.py     Progress + MemoryGuard: progress, RSS logging, budget enforcement
  artifacts.py    atomic JSON manifests for resumability
  store.py        TSV → normalised Parquet shards; RecordStore, PoolStore, SplitStore
  idmap.py        fixed-width sorted entity_id → rowid resolver (streaming build)
  indexes.py      build/save/load: inverted token (name, address), exact key,
                  sorted neighbourhood, country
  candidates.py   streaming per-batch candidate generation, union, dedup, cap
  recall.py       blocking recall against the verified pairs
scripts/
  build_store.py       CLI
  build_index.py       CLI
  smoke_candidates.py  CLI
tests/
  test_store.py        shard offset/order/pool-rowid correctness + id resolver
```

`normalize.py` from the original sketch is not a separate module: normalisation
lives in `store.build_table`, which reuses `src/preprocessing` in place so the
store stays the single definition of what "normalised" means for the project.

### Measured on the real train split

Run on 2026-09-27 against `dataset/student_resource/dataset/train/`, on a
20-core Windows box with 15.6 GB RAM. Budget was 6 GB RSS.

**Store** (`scripts/build_store.py --split train`), 12,526,040 rows:

| source | rows | shards | TSV | Parquet | peak RSS | time |
|--------|------|--------|-----|---------|----------|------|
| Source 1 | 2,206,821 | 9 | 200.3 MB | 179.3 MB | 1.6 GB | 3.1 min |
| Source 2 | 5,034,616 | 21 | 466.6 MB | 425.6 MB | 1.7 GB | 10.5 min |
| Source 3 | 5,285,603 | 22 | 480.4 MB | 440.2 MB | 1.7 GB | 10.5 min |
| **pool** | **10,320,219** | 43 | — | **865.8 MB** | — | — |

Total 24.2 min, peak 1.7 GB — 28% of the 6 GB budget, and flat in row count
because nothing is ever materialised whole.

**Index** (`scripts/build_index.py --split train`), 123.6 s, peak 3.2 GB,
1.9 GB on disk:

| index | size | notes |
|-------|------|-------|
| name token index | 286,673 terms / 10,872,912 postings / 168.3 MB | min_df=2, max_df=5000 |
| address token index | 535,600 terms / 23,072,038 postings / 267.0 MB | separate stop list |
| `name_norm` exact | 10,320,219 keys × 72 B / 803.5 MB | width auto-widened from 32 |
| `name_core` exact + SN | 10,320,219 keys × 64 B / 713.3 MB | width auto-widened from 32 |
| country | 2 codes / 39.4 MB | |
| pool id resolver | 10,320,219 ids × 12 B / 157.5 MB | sorted fixed-width, no dict |

The two exact-key indexes dominate memory because real business names exceed the
32-character design assumption; the store warns and widens rather than truncating,
since a truncated key would merge distinct names. This is the item to revisit
first if the budget ever gets tighter — hashing the keys to 8 bytes would cut
~1.5 GB at the cost of a collision check.

**Candidate generation** (`scripts/smoke_candidates.py --split train --limit 1000`),
first 1,000 real Source 1 rows against the full 10.3 M-row pool:

| metric | value |
|--------|-------|
| candidates | 99,151 (avg 99.15, max 100, 0 empty queries) |
| wall time | 2.2 s |
| throughput | 463 queries/s, 45,942 candidates/s |
| peak RSS | 629.5 MB (10% of budget) |
| **query recall** | **0.9419** — 891/946 labelled rows retrieved ≥1 true match |
| **pair recall** | **0.7728** — 2,718/3,517 verified pairs retrieved |
| all-pairs recall | 0.5307 — every partner found, which multi-match rows miss to the 100 cap |

54 of the first 1,000 Source 1 rows have an empty match list in
`train_ground_truth.tsv`; they are counted and excluded from all recall
denominators rather than scored as failures.

Candidate provenance (a candidate can be found by several strategies, so the
shares sum above 100%): name token 58.9%, address token 21.0%, exact core 19.7%,
sorted neighbourhood 14.9%, exact normalized 9.9%, country 2.2%.

Extrapolating the measured 463 queries/s, a full pass over all 2,206,821 Source 1
rows is ~80 min and yields ~220 M candidates (~5.5 GB of candidate Parquet).
The store and index are built and the candidate stage is measured, so that run is
a matter of starting it.

### Two bugs this stage's testing caught

Both were found by running against the real pool, not the 4,513-row fixture, and
both are now covered by `tests/test_store.py`:

1. `RecordStore.read_where` grouped rowids by shard but resolved the shard with
   `sorted_shards[group[0]]`, indexing the *sorted* array with a *group member*.
   Any gather spanning two or more shards opened the wrong file and computed a
   local offset that could be negative or past the shard end. On a single-shard
   fixture the distinction is invisible.
2. `PoolStore.read_where` wrote back the already-decremented Source 3 rowid, so
   the second half of the pool reported source-local rowids while documenting
   pool-wide ones.

The first one also made `pyarrow` raise on a perfectly valid gather, which is
how it surfaced. It is worth noting the failure mode was loud, not silent —
the shard-grouping bug could have returned wrong rows for a same-shard case
otherwise.

---

## Measured cap and quota experiment

`scripts/recall_experiment.py`, first 1,000 train Source 1 rows, full 10,320,219-row
pool, `train_ground_truth.tsv`. 946 rows have verified pairs (3,517 pairs); the other
54 are excluded from recall denominators. Reproduce with:

```
python scripts/recall_experiment.py --split train --limit 1000 \
    --json work/reports/recall_experiment.json
```

The run does **one uncapped generation pass** (1,627,151 candidates, mean 1,627,
p99 5,329, max 6,929, 1.8 s, 592 MB) and then slices it. A self-check asserts that
offline slicing reproduces a real `max_candidates_per_s1=100` run exactly — it does,
which is what makes the offline comparison trustworthy. Selection costs 0.03–0.19 s
for a 1.6 M-row union, so it is not a runtime concern; the cost of a larger cap is the
downstream candidate volume, not selection.

### The recall ceiling

With **no cap at all**, pair recall is **0.9446** (3,322 of 3,517 pairs) and query
recall 0.9926. Generation, not selection, sets the ceiling: 195 verified pairs are
retrieved by no strategy whatsoever. Fixing that is an index-coverage problem and is
independent of everything below.

### Per-strategy attribution (over the full uncapped union)

| strategy | candidates | share | pairs recovered | pairs found **only here** | pair recall if dropped |
| --- | ---: | ---: | ---: | ---: | ---: |
| exact name (normalized) | 12,309 | 0.7% | 987 | 0 | 1.0000 |
| exact name (legal-form stripped) | 40,474 | 2.4% | 1,707 | 37 | 0.9895 |
| name token | 754,785 | 45.4% | 2,000 | 88 | 0.9750 |
| character/ngram (sorted neighbourhood) | 21,283 | 1.3% | 817 | 9 | 0.9974 |
| **address token** | 784,014 | 47.1% | 2,831 | **807** | **0.7705** |
| country | 50,000 | 3.0% | **0** | **0** | 1.0000 |

Two results drive every decision that follows:

- **Address is the load-bearing strategy.** 807 of 3,517 pairs (23.0%) are found by
  address and by nothing else. Dropping it costs 0.174 pair recall — more than every
  other strategy combined.
- **Country contributes nothing.** 50,000 candidates, 0 verified pairs recovered, 0
  exclusive. Every country hit was also found by some other strategy, so the country
  channel is pure filler. It is worth keeping as a *scoring* feature — the module
  docstring already says so — but it should not be a *retrieval* strategy.

### Why the cap hurts, and who it hurts

| sole finder of the pair | pairs | median global rank | p90 | kept @100 | @200 | @500 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| multiple strategies | 2,381 | 2 | 24 | 2,349 | 2,365 | 2,374 |
| address token | 807 | 172 | 2,164 | 282 | 427 | 554 |
| name token | 88 | 76 | 766 | 48 | 55 | 73 |
| exact core | 37 | 8 | 108 | 33 | 35 | 37 |
| sorted neighbourhood | 9 | 15 | 1,989 | 6 | 6 | 6 |

Pairs found by several strategies survive any reasonable cap (median rank 2). The
loss is concentrated almost entirely in **address-only** pairs: 525 of the 604 pairs
lost at cap 100 are address-only. The cause is structural, not incidental —
`rule_score` is the *sum* of strategy weights, so a name-only hit (10) outranks an
address-only hit (4) no matter how strong the address match is, and with ~750 name
candidates per query an address-only true match cannot surface until deep down the
list.

### Cap sweep

| cap | candidates | avg | query recall | pair recall | all-pairs recall |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 50 | 50,000 | 50.0 | 0.9038 | 0.7117 | 0.4577 |
| 100 | 99,151 | 99.2 | 0.9419 | 0.7728 | 0.5307 |
| 150 | 146,952 | 146.9 | 0.9588 | 0.8069 | 0.5729 |
| 200 | 193,279 | 193.3 | 0.9619 | 0.8212 | 0.5973 |
| 300 | 282,501 | 282.5 | 0.9704 | 0.8411 | 0.6290 |
| 500 | 446,249 | 446.2 | 0.9767 | 0.8655 | 0.6734 |

### Per-strategy quotas: the cap mode is the whole story

`select_with_quotas` has two cap modes, and the difference decides whether quotas
mean anything:

- **`rank`** — take the global top *N* first, then intersect with the quota union.
  Because name candidates dominate the top of the ranking, the cap is already full
  before any address quota is consulted, so **the quotas are inert**.
- **`union`** — form the quota union first, then trim *it* to the cap by rank. Each
  quota is a floor that survives the trim. This is the only mode in which a
  per-strategy budget can redirect slots.

Measured (all at the stated cap, country-only and exact-normalized-only candidates
excluded):

| configuration | mode | candidates | query recall | pair recall | all-pairs recall |
| --- | --- | ---: | ---: | ---: | ---: |
| cap 100 (baseline) | — | 99,151 | 0.9419 | 0.7728 | 0.5307 |
| C: 50 name / 30 addr / 20 char / 10 country | rank | 58,847 | 0.9376 | 0.7572 | 0.5021 |
| C2: same quotas | **union** | 81,784 | 0.9471 | **0.8007** | **0.5930** |
| G2: measured quota (18/56/6) | **union** | 71,641 | 0.9514 | **0.8081** | 0.6004 |
| D2: adaptive on exact-name signal | **union** | 87,387 | 0.9535 | 0.8081 | 0.5941 |
| cap 200 | — | 193,279 | 0.9619 | 0.8212 | 0.5973 |
| H2: measured quota (30/100/14) | **union** | 124,597 | 0.9619 | **0.8379** | **0.6448** |
| I2: address-heavy (20/150/20) | **union** | 154,671 | 0.9651 | **0.8450** | 0.6543 |
| J2: address-heavier (10/170/20) | **union** | 163,291 | 0.9651 | **0.8465** | 0.6554 |
| K2: address-heavy at cap 150 (10/120/20) | **union** | 123,647 | 0.9609 | 0.8334 | 0.6342 |
| L2: address-heavy at cap 250 (20/200/30) | **union** | 196,034 | 0.9725 | **0.8607** | **0.6786** |
| M2: name-heavy control (100/70/30) | **union** | 142,854 | 0.9619 | 0.8359 | 0.6501 |
| cap 300 | — | 282,501 | 0.9704 | 0.8411 | 0.6290 |
| cap 500 | — | 446,249 | 0.9767 | 0.8655 | 0.6734 |

The name-heavy control (M2) and the address-heavy variants (I2/J2) differ by 0.010
pair recall at the same cap, which is the attribution showing up in the result: slots
are worth more given to address than to name.

### Efficiency frontier

For each configuration, the cheapest plain cap that reaches the same pair recall:

| configuration | candidates | pair recall | all-pairs | cheapest equal plain cap | volume saved |
| --- | ---: | ---: | ---: | --- | ---: |
| L2 address-heavy cap 250 | 196,034 | 0.8607 | 0.6786 | cap 500 (446,249) | **56.1%** |
| J2 address-heavy cap 200 | 163,291 | 0.8465 | 0.6554 | cap 500 (446,249) | **63.4%** |
| H2 measured quota cap 200 | 124,597 | 0.8379 | 0.6448 | cap 300 (282,501) | **55.9%** |
| K2 address-heavy cap 150 | 123,647 | 0.8334 | 0.6342 | cap 300 (282,501) | **56.2%** |
| G2 measured quota cap 100 | 71,641 | 0.8081 | 0.6004 | cap 200 (193,279) | **62.9%** |
| C2 (the 50/30/20 proposal) | 81,784 | 0.8007 | 0.5930 | cap 150 (146,952) | 44.3% |
| C (same, rank mode) | 58,847 | 0.7572 | 0.5021 | cap 100 (99,151) | — (worse recall) |

### Recommendation

1. **Use union-mode quotas, not a plain cap.** The single global rank is dominated at
   every volume level: 55–65% fewer candidates for equal-or-better recall.
2. **Suggested setting: cap 200, `name_token` 20 / `address_token` 150 /
   `sorted_neighbourhood` 30, exact strategies unlimited, country-only and
   exact-normalized-only candidates dropped** (configuration I2). 0.8450 pair recall
   and 0.6543 all-pairs at 155 candidates/query, versus 0.8212 / 0.5973 at 193 for
   plain cap 200 and 0.8411 / 0.6290 at 283 for plain cap 300.
3. **If recall matters more than volume, cap 250 with 20/200/30** (L2): 0.8607 /
   0.6786 at 196 candidates/query — still 56% cheaper than plain cap 500 and with
   *better* all-pairs recall than it.
4. **Retire country as a retrieval strategy**; keep the bit as a scoring feature. It
   measured 0 pairs over 50,000 candidates.
5. **Do not adopt the adaptive exact-name rule.** D2 matches G2's pair recall exactly
   (0.8081) while using 22% more candidates and 2.4× the selection time, and its
   rank-mode twin is the worst configuration measured. `exact_normalized` has 0
   exclusive pairs and `exact_core` only 37, so the exact signal is not a useful
   trigger for reallocating slots — the static measured quota captures the entire
   benefit.
6. **The remaining gap is address-side, not cap-side.** 195 pairs are unreachable by
   any strategy (ceiling 0.9446) and 253 address-only pairs are still lost even at
   cap 500. Both need address index work — tighter posting lists or a minimum
   similarity floor that promotes genuine address matches — not a bigger cap or a
   different allocation.

### Bugs this experiment's self-checks and tests caught

All are covered by `tests/test_candidates.py` (20 new tests; suite is 65 passing).

1. **A cap with no quotas silently meant "exact hits only."** The `always` list was
   applied even when `quotas` was empty, so a cap of 100 kept the 100 best *exact*
   matches and dropped every name, address and character candidate whenever the query
   had any exact hit. This is the single worst bug in the change: it made the
   baseline appear to work while quietly discarding 99% of the candidate set.
2. **`exclude_only` returned early and discarded `quotas`.** Every configuration that
   combined the two behaved as if the quotas had never been evaluated, which reads as
   "quotas do nothing" — the first reason the quota experiment appeared to fail.
3. **Exclusion counted excluded rows against the cap.** Dropping a strategy deleted
   those rows and never backfilled, so the cap quietly shrank instead of handing the
   freed budget to the next best candidates.
4. **A subset-aware rank helper indexed group bases by query value instead of group
   start**, so ranks leaked across query boundaries. Found by a test that compares the
   vectorised rank against an explicit Python loop.
5. **A diagnostic permuted by `pool_rowid` before ranking.** The uncapped batch is
   ordered by score, not by rowid, so the reported ranks were scrambled and the
   diagnostic disagreed with the measured cap-100 recall (354 pairs vs 2,718). The
   strategy attribution was unaffected — it searched and indexed in the same permuted
   space — which is why only the rank diagnostic was wrong.

The rank diagnostic and the offline-slicing self-check are the two pieces that made
the rest trustworthy: one caught a permuted index, the other caught bug 1 above. Both
compare against a brute-force computation rather than a previously recorded number.


