"""Unit tests for the F0.5 metric.

The expected values are computed by hand from the official formula::

    F0.5 = 1.25 * precision * recall / (0.25 * precision + recall)

so a regression in the implementation cannot be masked by the implementation
itself.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.metrics import (
    F05Evaluator,
    best_threshold,
    f05_from_counts,
    f05_per_entity,
    f05_score,
    summarize_confusion,
)

# (gt, pred, expected_f05)
CASES = [
    # singleton handled correctly -> full credit
    ((), (), 1.0),
    # singleton incorrectly matched -> zero
    ((), ("m1",), 0.0),
    (("m1",), (), 0.0),
    # perfect single match
    (("m1",), ("m1",), 1.0),
    # one true, one false: p=1/2, r=1 -> 1.25*0.5*1/(0.25*0.5+1) = 0.625/1.125
    (("m1",), ("m1", "m2"), 0.625 / 1.125),
    # wrong match only
    (("m1",), ("m2",), 0.0),
    # two true, one found: p=1, r=1/2 -> 1.25*0.5/(0.25+0.5)
    (("m1", "m2"), ("m1",), 1.25 * 0.5 / (0.25 + 0.5)),
    # two true, both found -> 1.0
    (("m1", "m2"), ("m1", "m2"), 1.0),
    # two true, both found plus one false: p=2/3, r=1
    (("m1", "m2"), ("m1", "m2", "m3"), 1.25 * (2 / 3) * 1.0 / (0.25 * (2 / 3) + 1.0)),
    # three true, one found: p=1, r=1/3
    (("a", "b", "c"), ("a",), 1.25 * (1 / 3) / (0.25 + 1 / 3)),
]


@pytest.mark.parametrize("gt,pred,expected", CASES)
def test_f05_per_entity(gt, pred, expected):
    assert f05_per_entity(gt, pred) == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("gt,pred,expected", CASES)
def test_f05_from_counts_matches_reference(gt, pred, expected):
    tp = len(set(gt) & set(pred))
    assert f05_from_counts(tp, len(set(pred)), len(set(gt))) == pytest.approx(expected, abs=1e-12)


def test_macro_average_over_entities():
    gt = {"a": ("m1",), "b": (), "c": ("n1", "n2")}
    pred = {"a": ("m1",), "b": (), "c": ("n1",)}
    # a: perfect -> 1.0 ; b: correctly unmatched -> 1.0 ;
    # c: p=1, r=1/2 -> 1.25*0.5 / (0.25*1 + 0.5)
    expected = (1.0 + 1.0 + (1.25 * 0.5 / (0.25 + 0.5))) / 3
    assert f05_score(gt, pred) == pytest.approx(expected, abs=1e-12)


def test_missing_prediction_key_is_treated_as_empty():
    gt = {"a": ("m1",), "b": ()}
    assert f05_score(gt, {"a": ("m1",)}) == pytest.approx(1.0)


def _synthetic():
    """Two entities: 'a' has 2 truths, 'b' has none."""
    truth = {"a": frozenset({"m1", "m2"}), "b": frozenset()}
    entities = ["a", "b"]
    pairs = [
        (0, "m1", 0.95, 1),
        (0, "m2", 0.90, 1),
        (0, "x1", 0.60, 0),
        (1, "y1", 0.55, 0),
    ]
    ent = np.array([p[0] for p in pairs])
    mid = [p[1] for p in pairs]
    sc = np.array([p[2] for p in pairs])
    y = np.array([p[3] for p in pairs])
    return F05Evaluator(entities, truth), ent, mid, sc, y


def test_evaluator_matches_reference_implementation():
    ev, ent, mid, sc, _ = _synthetic()
    for t in np.arange(0.3, 1.0, 0.05):
        pred = ev.predictions_from_scores(ent, mid, sc, t)
        reference = f05_score({e: ev.truth[e] for e in ev.entity_ids}, pred)
        point = ev.evaluate_threshold(ent, mid, sc, float(t))
        assert point.f05 == pytest.approx(reference, abs=1e-12), f"threshold {t}"


def test_singleton_violation_counts():
    ev, ent, mid, sc, _ = _synthetic()
    point = ev.evaluate_threshold(ent, mid, sc, 0.5)
    # entity 'a' -> m1, m2, x1 ; entity 'b' -> y1 (a singleton violation)
    assert point.tp == 2
    assert point.fp == 2
    assert point.fn == 0
    assert point.n_singleton_violations == 1
    expected = (f05_per_entity({"m1", "m2"}, {"m1", "m2", "x1"}) + f05_per_entity((), {"y1"})) / 2
    assert point.f05 == pytest.approx(expected, abs=1e-12)


def test_sweep_finds_the_precision_optimum():
    ev, ent, mid, sc, _ = _synthetic()
    grid = ev.sweep(ent, mid, sc, np.arange(0.30, 0.96, 0.05))
    best = best_threshold(grid)
    # t<=0.55 -> 'a' is perfect but 'b' is violated; t in (0.60, 0.90] -> both fine
    assert best.threshold == pytest.approx(0.90)
    assert best.f05 == pytest.approx(1.0)


def test_breakpoint_sweep_matches_grid_sweep_on_reachable_values():
    ev, ent, mid, sc, _ = _synthetic()
    points = ev.sweep_all_breakpoints(ent, mid, sc, max_points=100)
    assert len(points) == len({0.95, 0.90, 0.60, 0.55})
    assert max(p.f05 for p in points) == pytest.approx(1.0)
    # The global optimum over all breakpoints equals the optimum over a fine grid.
    grid = ev.sweep(ent, mid, sc, np.linspace(0.0, 1.0, 2001))
    assert best_threshold(points).f05 == pytest.approx(best_threshold(grid).f05, abs=1e-12)


def test_f05_favours_precision_over_recall():
    """An operating point that adds one false positive must lose."""
    ev = F05Evaluator(["a"], {"a": frozenset({"m1"})})
    ent = np.array([0, 0])
    mid = ["m1", "m2"]
    sc = np.array([0.9, 0.85])
    good = ev.evaluate_threshold(ent, mid, sc, 0.9)
    bad = ev.evaluate_threshold(ent, mid, sc, 0.85)
    assert good.f05 == 1.0
    assert bad.f05 < good.f05
    assert best_threshold([bad, good]).threshold == pytest.approx(0.9)


def test_confusion_summary_renders():
    ev, ent, mid, sc, _ = _synthetic()
    text = summarize_confusion(ev.sweep(ent, mid, sc, [0.5, 0.9]))
    assert "F0.5" in text and "singletonFP" in text


# ---------------------------------------------------------------------------
# Tied scores
#
# Regression tests for a bug where the breakpoint sweep recorded an operating
# point after only the *first* pair of a group of equal scores had been
# counted.  At threshold t every pair scoring t is predicted, so the running
# counts have to advance over the whole tie group first.  The old code
# under-counted n_pred/tp at each breakpoint, which inflated precision and
# reported F0.5 = 0.87 where the true value was 0.27.
# ---------------------------------------------------------------------------


def _tied_case():
    """Heavy score ties across several entities, including true singletons."""
    truth = {
        "a": frozenset({"m1", "m2"}),
        "b": frozenset({"m3"}),
        "c": frozenset(),          # singleton: predicting anything scores 0
        "d": frozenset({"m4"}),
    }
    ev = F05Evaluator(sorted(truth), truth)
    ent = np.array([0, 0, 1, 2, 2, 3])
    mid = ["m1", "m2", "m3", "m5", "m6", "m4"]
    sc = np.array([0.50, 0.50, 0.50, 0.50, 0.50, 0.50])  # every score identical
    return ev, ent, mid, sc, truth


def test_breakpoint_sweep_agrees_with_direct_evaluation_when_scores_tie():
    ev, ent, mid, sc, _ = _tied_case()
    points = ev.sweep_all_breakpoints(ent, mid, sc, max_points=100)
    assert len(points) == 1
    direct = ev.evaluate_threshold(ent, mid, sc, 0.50)
    assert points[0].f05 == pytest.approx(direct.f05, abs=1e-12)
    assert points[0].n_pred_pairs == direct.n_pred_pairs == 6
    # entity 'a' -> 2/2 correct, 'b' -> 1/1, 'd' -> 1/1, 'c' violated (0)
    assert points[0].f05 == pytest.approx(0.75, abs=1e-12)


def test_breakpoint_sweep_counts_whole_tie_group():
    ev, ent, mid, sc, _ = _tied_case()
    # Above the tie nothing is predicted: 'a', 'b' and 'd' each have a true
    # match and score 0, while the true singleton 'c' is satisfied (1.0).
    nothing = ev.evaluate_threshold(ent, mid, sc, 0.51)
    assert nothing.n_pred_pairs == 0
    assert nothing.f05 == pytest.approx(0.25, abs=1e-12)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_breakpoint_sweep_matches_brute_force_macro(seed):
    """Every breakpoint must equal the reference macro average at that score."""
    rng = np.random.default_rng(seed)
    n_entities = 40
    truth = {}
    ent, mid, sc = [], [], []
    for i in range(n_entities):
        eid = f"e{i}"
        n_true = int(rng.integers(0, 3))
        matches = {f"t{i}_{k}" for k in range(n_true)}
        truth[eid] = frozenset(matches)
        for k in range(int(rng.integers(1, 6))):
            ent.append(i)
            # quantised scores -> plenty of ties
            sc.append(float(rng.integers(0, 11)) / 10.0)
            mid.append(f"t{i}_{k}" if (matches and rng.random() < 0.6) else f"f{i}_{k}")
    ev = F05Evaluator([f"e{i}" for i in range(n_entities)], truth)
    ent_a = np.array(ent, dtype=np.int64)
    sc_a = np.array(sc, dtype=np.float64)
    for point in ev.sweep_all_breakpoints(ent_a, mid, sc_a, max_points=500):
        pred: dict[str, set[str]] = {}
        for idx in range(len(ent_a)):
            if sc_a[idx] >= point.threshold:
                pred.setdefault(ev.entity_ids[int(ent_a[idx])], set()).add(mid[idx])
        brute = f05_score(truth, pred)
        assert point.f05 == pytest.approx(brute, abs=1e-12), (
            f"seed={seed} threshold={point.threshold}: {point.f05} != {brute}"
        )


def test_optimized_threshold_never_exceeds_true_achievable_f05():
    """The reported optimum must be reproducible by a brute-force sweep."""
    ev, ent, mid, sc, truth = _tied_case()
    from src.evaluation.metrics import best_threshold as _best

    points = ev.sweep_all_breakpoints(ent, mid, sc, max_points=500)
    chosen = _best(points)
    pred: dict[str, set[str]] = {}
    for idx in range(len(ent)):
        if sc[idx] >= chosen.threshold:
            pred.setdefault(ev.entity_ids[int(ent[idx])], set()).add(mid[idx])
    assert chosen.f05 == pytest.approx(f05_score(truth, pred), abs=1e-12)
