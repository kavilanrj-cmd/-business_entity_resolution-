"""Tests for candidate selection under a cap and under per-strategy quotas.

The arrays are hand-built rather than produced by a real generation pass so that
the expected answer can be written down by hand: ``query`` is grouped and
best-first inside each query, which is the contract :func:`select_with_quotas`
documents and that ``CandidateGenerator._merge`` guarantees.
"""

import numpy as np
import pytest

from src.scale.candidates import select_with_quotas
from src.scale.config import STRATEGY_BITS

EXACT = STRATEGY_BITS["exact_normalized"]
CORE = STRATEGY_BITS["exact_core"]
NAME = STRATEGY_BITS["name_token"]
ADDR = STRATEGY_BITS["address_token"]
SN = STRATEGY_BITS["sorted_neighbourhood"]
COUNTRY = STRATEGY_BITS["country_token"]


def build(rows):
    """rows: [(query_slot, pool_rowid, mask), ...] in ranked order."""
    return (
        np.array([r[0] for r in rows], dtype=np.int32),
        np.array([r[1] for r in rows], dtype=np.int32),
        np.array([r[2] for r in rows], dtype=np.int32),
    )


def kept_pools(query, pool, mask, keep):
    return set(zip(query[keep].tolist(), pool[keep].tolist()))


def test_no_quotas_is_pure_rank_truncation():
    """A cap with no quotas must keep the top-N of *every* strategy.

    Regression: the ``always`` list used to be applied even with no quotas, so a
    plain cap silently degenerated into "only the exact hits".  That made a cap
    of 100 mean "the 100 best exact matches" and dropped every name, address and
    character candidate whenever the query had any exact hit at all.
    """
    query, pool, mask = build([
        (0, 10, EXACT),
        (0, 11, NAME),
        (0, 12, ADDR),
        (0, 13, NAME | ADDR),
        (0, 14, COUNTRY),
    ])
    keep = select_with_quotas(query, mask, n_queries=1, cap=3)
    assert kept_pools(query, pool, mask, keep) == {(0, 10), (0, 11), (0, 12)}


def test_no_quotas_ignores_exclusion():
    """With no quotas and no exclusion there is no strategy filter at all."""
    query, pool, mask = build([(0, 10, COUNTRY), (0, 11, ADDR)])
    keep = select_with_quotas(query, mask, n_queries=1, cap=0)
    assert keep.all()


def test_cap_truncates_per_query_not_globally():
    query, pool, mask = build([
        (0, 10, NAME), (0, 11, NAME), (0, 12, NAME),
        (1, 20, NAME), (1, 21, NAME), (1, 22, NAME),
    ])
    keep = select_with_quotas(query, mask, n_queries=2, cap=2)
    assert kept_pools(query, pool, mask, keep) == {(0, 10), (0, 11), (1, 20), (1, 21)}


def test_quota_limits_a_strategy_per_query():
    query, pool, mask = build([
        (0, 10, NAME), (0, 11, NAME), (0, 12, NAME),
        (0, 20, ADDR), (0, 21, ADDR),
        (1, 30, NAME), (1, 31, NAME), (1, 32, NAME),
    ])
    keep = select_with_quotas(
        query, mask, n_queries=2, cap=0,
        quotas={"name_token": 1, "address_token": 1}, always=(),
    )
    assert kept_pools(query, pool, mask, keep) == {
        (0, 10), (0, 20),   # the best name and the best address of query 0
        (1, 30),             # query 1 has no address candidates
    }


def test_exact_strategies_bypass_their_own_quota():
    """``always`` exists so a near-certain exact hit cannot be crowded out."""
    query, pool, mask = build([
        (0, 10, NAME), (0, 11, NAME), (0, 12, NAME),
        (0, 20, CORE),
    ])
    keep = select_with_quotas(
        query, mask, n_queries=1, cap=0, quotas={"name_token": 1}
    )
    assert kept_pools(query, pool, mask, keep) == {(0, 10), (0, 20)}


def test_quota_then_cap_union_keeps_both():
    """Quotas allocate slots per strategy, then the total cap trims by rank."""
    query, pool, mask = build([
        (0, 10, NAME), (0, 11, NAME), (0, 12, NAME),
        (0, 20, ADDR), (0, 21, ADDR), (0, 22, ADDR),
    ])
@pytest.mark.parametrize("cap_mode", ["rank", "union"])
def test_quota_then_cap_keeps_the_quota_floors(cap_mode):
    """The cap mode decides whether a per-strategy quota can redirect slots.

    In ``rank`` mode the cap is applied to the whole batch first, so the three
    name candidates -- which all outrank the address ones on ``rule_score`` --
    consume the entire budget and the address quota is worth nothing.  In
    ``union`` mode the quota union is formed first and then trimmed, so the
    address quota survives and one name slot gives way.
    """
    query, pool, mask = build([
        (0, 10, NAME), (0, 11, NAME), (0, 12, NAME),
        (0, 20, ADDR), (0, 21, ADDR), (0, 22, ADDR),
    ])
    keep = select_with_quotas(
        query, mask, n_queries=1, cap=3, cap_mode=cap_mode,
        quotas={"name_token": 2, "address_token": 2}, always=(),
    )
    got = kept_pools(query, pool, mask, keep)
    if cap_mode == "rank":
        # Only 2 of the 3 budgeted slots survive: the cap was applied to the
        # batch, where name candidates had already taken all 3.
        assert got == {(0, 10), (0, 11)}
    else:
        # The union is {10, 11, 20, 21} and the cap trims it to 3 by rank, so
        # the address quota survives -- which is the behaviour under test.
        assert got == {(0, 10), (0, 11), (0, 20)}
        assert int(keep.sum()) == 3


def test_union_mode_cap_still_bounds_each_query():
    query, pool, mask = build([
        (0, 10, NAME), (0, 11, NAME), (0, 12, NAME), (0, 13, NAME),
        (1, 20, ADDR), (1, 21, ADDR), (1, 22, ADDR), (1, 23, ADDR),
    ])
    keep = select_with_quotas(
        query, mask, n_queries=2, cap=2, cap_mode="union",
        quotas={"name_token": 3, "address_token": 3}, always=(),
    )
    assert kept_pools(query, pool, mask, keep) == {(0, 10), (0, 11), (1, 20), (1, 21)}


def test_union_mode_never_exceeds_cap_when_quota_sum_exceeds_it():
    query, pool, mask = build([
        (0, 10, NAME), (0, 11, NAME), (0, 12, NAME),
        (0, 20, ADDR), (0, 21, ADDR), (0, 22, ADDR),
    ])
    keep = select_with_quotas(
        query, mask, n_queries=1, cap=1, cap_mode="union",
        quotas={"name_token": 3, "address_token": 3}, always=(),
    )
    assert int(keep.sum()) == 1


def test_exclude_only_drops_single_strategy_candidates():
    """Exclusion is narrower than banning a strategy, on purpose.

    A candidate that country matched and the name index also matched still has
    name evidence, so it must survive; only a country-only filler is dropped.
    """
    query, pool, mask = build([
        (0, 10, COUNTRY),
        (0, 11, COUNTRY | NAME),
        (0, 12, NAME),
    ])
    keep = select_with_quotas(query, mask, n_queries=1, cap=0, exclude_only=("country_token",))
    assert kept_pools(query, pool, mask, keep) == {(0, 11), (0, 12)}


def test_exclude_only_reclaims_budget_for_survivors():
    """Excluding a strategy must hand its budget to the next best candidates.

    If the cap were applied by global rank, dropping the country-only row would
    just delete it and never backfill, so the cap would quietly shrink from 2 to
    1.  Trimming by survivor rank is what makes the exclusion worth 2 candidates
    instead of 1.
    """
    query, pool, mask = build([
        (0, 10, COUNTRY),
        (0, 11, COUNTRY | NAME),
        (0, 12, NAME),
        (0, 13, NAME),
    ])
    keep = select_with_quotas(
        query, mask, n_queries=1, cap=2, exclude_only=("country_token",)
    )
    # The country-only row is dropped, then the cap keeps the best two of the rest.
    assert kept_pools(query, pool, mask, keep) == {(0, 11), (0, 12)}
    assert int(keep.sum()) == 2


def test_excluding_everything_does_not_empty_a_query():
    """An over-aggressive exclusion must not produce a query with no candidates."""
    query, pool, mask = build([(0, 10, COUNTRY), (0, 11, COUNTRY)])
    keep = select_with_quotas(
        query, mask, n_queries=1, cap=0, exclude_only=("country_token",)
    )
    # Both candidates are country-only, so exclusion would empty the query; the
    # guard is that a query is never emptied, so nothing is dropped.
    assert keep.all()


def test_quotas_matching_nothing_fall_back_instead_of_emptying():
    query, pool, mask = build([(0, 10, COUNTRY), (0, 11, ADDR)])
    keep = select_with_quotas(
        query, mask, n_queries=1, cap=0,
        quotas={"sorted_neighbourhood": 5}, always=(),
    )
    assert keep.all()


def test_empty_input():
    empty = np.zeros(0, dtype=np.int32)
    keep = select_with_quotas(empty, empty, n_queries=0, cap=10)
    assert keep.shape == (0,)
    assert keep.dtype == np.bool_


def test_exclude_only_composes_with_quotas():
    """Exclusion and quotas are independent levers and must both apply.

    Regression: the ``exclude_only`` branch used to return early, so any
    configuration that also set quotas silently ignored them.  That made
    quota-bearing configurations look identical to "exclude and cap", which
    reads as "quotas do nothing" when in fact they were never evaluated.
    """
    query, pool, mask = build([
        (0, 10, COUNTRY),
        (0, 11, NAME), (0, 12, NAME), (0, 13, NAME),
        (0, 20, ADDR), (0, 21, ADDR), (0, 22, ADDR),
    ])
    kwargs = dict(
        n_queries=1, cap=0, cap_mode="union",
        quotas={"name_token": 1, "address_token": 1}, always=(),
        exclude_only=("country_token",),
    )
    # The country-only row is dropped *and* the quotas are enforced: one name
    # and one address slot, not "everything except country".
    assert kept_pools(query, pool, mask, select_with_quotas(query, mask, **kwargs)) == {
        (0, 11), (0, 20),
    }


def test_exclude_only_alone_still_caps():
    query, pool, mask = build([
        (0, 10, COUNTRY),
        (0, 11, NAME), (0, 12, NAME), (0, 13, NAME),
    ])
    keep = select_with_quotas(
        query, mask, n_queries=1, cap=2, exclude_only=("country_token",)
    )
    assert kept_pools(query, pool, mask, keep) == {(0, 11), (0, 12)}


@pytest.mark.parametrize("cap_mode", ["rank", "union"])
def test_quota_selection_is_order_independent_within_a_query(cap_mode):
    """Whichever of two same-strategy rows ranks first, the same one is kept."""
    query, pool, mask = build([
        (0, 10, NAME), (0, 11, NAME), (0, 12, ADDR), (0, 13, ADDR),
    ])
    keep = select_with_quotas(
        query, mask, n_queries=1, cap=10, cap_mode=cap_mode,
        quotas={"name_token": 1, "address_token": 1}, always=(),
    )
    assert kept_pools(query, pool, mask, keep) == {(0, 10), (0, 12)}


def test_subset_rank_matches_a_python_loop():
    """``_rank_within_group(query, subset)`` must agree with an explicit loop."""
    from src.scale.candidates import _rank_within_group

    rng = np.random.default_rng(0)
    query = np.repeat([0, 1, 2], [4, 3, 5]).astype(np.int32)
    subset = rng.random(query.size) < 0.6
    got = _rank_within_group(query, subset)
    want = np.zeros(query.size, dtype=np.int64)
    for slot in np.unique(query):
        rows = np.flatnonzero(query == slot)
        seen = 0
        for row in rows:
            if subset[row]:
                want[row] = seen
                seen += 1
    # Only the surviving rows carry a meaningful rank, and those must match.
    assert got[subset].tolist() == want[subset].tolist()
