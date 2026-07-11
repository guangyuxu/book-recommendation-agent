"""Scoring primitives for the eval strategies.

Kept generic and dependency-free (no sklearn/numpy): the eval harness must run in a bare
CI image. S1 uses all three functions here; S2/S3 will add their own aggregations later.

Intent classification is a MULTI-LABEL problem (a turn may carry several intents), so we report
two complementary views:
- `set_exact_match`: the strict, product-meaningful metric -- did we predict EXACTLY the right
  set (no misses, no extras)? This is what a "the turn was understood correctly" bar looks like.
- `label_prf`: per-label precision/recall/F1 plus micro/macro rollups -- the diagnostic view that
  tells you WHICH intents are weak when the exact-match rate drops.
"""

from __future__ import annotations

from collections.abc import Sequence


def set_exact_match(
    pred_sets: Sequence[set[str]], gold_sets: Sequence[set[str]]
) -> float:
    """Fraction of cases where the predicted label set equals the gold set exactly.

    Returns 0.0 for an empty dataset (nothing correct rather than a divide-by-zero).
    """
    if not gold_sets:
        return 0.0
    hits = sum(1 for p, g in zip(pred_sets, gold_sets) if p == g)
    return hits / len(gold_sets)


def _prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    """Precision/recall/F1 from raw counts; each is 0.0 when its denominator is 0."""
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def label_prf(
    pred_sets: Sequence[set[str]],
    gold_sets: Sequence[set[str]],
    labels: Sequence[str],
) -> dict:
    """Per-label precision/recall/F1 with micro and macro rollups.

    micro_f1 pools TP/FP/FN across all labels (dominated by frequent labels); macro_f1 averages
    each label's F1 equally (surfaces weak rare labels). `per_label[label].support` is the gold
    count, so you can tell a genuinely-hard label from one with too few examples to trust.
    """
    per_label: dict[str, dict[str, float]] = {}
    tot_tp = tot_fp = tot_fn = 0
    for label in labels:
        tp = sum(1 for p, g in zip(pred_sets, gold_sets) if label in p and label in g)
        fp = sum(
            1 for p, g in zip(pred_sets, gold_sets) if label in p and label not in g
        )
        fn = sum(
            1 for p, g in zip(pred_sets, gold_sets) if label not in p and label in g
        )
        support = sum(1 for g in gold_sets if label in g)
        scores = _prf(tp, fp, fn)
        per_label[label] = {**scores, "support": float(support)}
        tot_tp += tp
        tot_fp += fp
        tot_fn += fn

    micro = _prf(tot_tp, tot_fp, tot_fn)
    macro_f1 = (
        sum(s["f1"] for s in per_label.values()) / len(per_label) if per_label else 0.0
    )
    return {
        "micro_precision": micro["precision"],
        "micro_recall": micro["recall"],
        "micro_f1": micro["f1"],
        "macro_f1": macro_f1,
        "per_label": per_label,
    }


def resolution_accuracy(
    pred_ids: Sequence[str | None], gold_ids: Sequence[str | None]
) -> float:
    """Fraction of cases where the resolved target_child_id matches the gold id (None == None ok).

    Returns 0.0 for an empty dataset.
    """
    if not gold_ids:
        return 0.0
    hits = sum(1 for p, g in zip(pred_ids, gold_ids) if p == g)
    return hits / len(gold_ids)


# --- judge aggregation (S = judge strategy) --------------------------------------------
#
# A judge run yields one integer score per rubric DIMENSION per case (see `_harness/judge.py`).
# These two helpers roll a list of per-case score dicts up into the flat `summary` metrics that
# thresholds gate on. Kept dependency-free, same as the classification metrics above.


def mean_by_dimension(
    scores: Sequence[dict[str, int]], dimensions: Sequence[str]
) -> dict[str, float]:
    """Mean score per dimension across cases, keyed `mean_<dimension>`.

    A missing dimension in a case (e.g. the judge failed on that case) is skipped rather than
    counted as 0, so one bad case lowers the sample size, not the average unfairly. A dimension
    with no valid scores at all reports 0.0.
    """
    out: dict[str, float] = {}
    for dim in dimensions:
        vals = [s[dim] for s in scores if isinstance(s.get(dim), (int, float))]
        out[f"mean_{dim}"] = sum(vals) / len(vals) if vals else 0.0
    return out


def pass_rate(
    scores: Sequence[dict[str, int]], dimensions: Sequence[str], floor: int
) -> float:
    """Fraction of cases where EVERY dimension scored >= `floor` (the product 'good enough' bar).

    Complements the per-dimension means: a booklist that is great on two dimensions and terrible
    on the third averages to 'fine' but fails here, which is usually what you actually care about.
    Returns 0.0 for an empty dataset.
    """
    if not scores:
        return 0.0
    ok = sum(1 for s in scores if all(s.get(d, 0) >= floor for d in dimensions))
    return ok / len(scores)
