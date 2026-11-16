
from collections import defaultdict


METRICS = (
    "tool_call_accuracy",
    "tool_call_f1",
)


def index_results(records):
    indexed = {}

    for record in records:
        test_id = record["test_id"]

        if test_id in indexed:
            raise ValueError(
                f"Duplicate test ID in evaluation run: {test_id}"
            )

        indexed[test_id] = record

    return indexed


def compare_evaluation_runs(
    baseline_records,
    candidate_records,
    tolerance=0.0,
):
    if not 0.0 <= tolerance <= 1.0:
        raise ValueError("Tolerance must be between 0 and 1")

    if not baseline_records or not candidate_records:
        raise ValueError("Evaluation runs cannot be empty")

    baseline = index_results(baseline_records)
    candidate = index_results(candidate_records)

    if set(baseline) != set(candidate):
        missing = sorted(set(baseline) - set(candidate))
        extra = sorted(set(candidate) - set(baseline))

        raise ValueError(
            f"Test case mismatch: missing={missing}, extra={extra}"
        )

    baseline_hashes = {
        r["dataset_sha256"] for r in baseline_records
    }
    candidate_hashes = {
        r["dataset_sha256"] for r in candidate_records
    }

    if len(baseline_hashes) != 1:
        raise ValueError("Baseline contains multiple dataset hashes")

    if len(candidate_hashes) != 1:
        raise ValueError("Candidate contains multiple dataset hashes")

    if baseline_hashes != candidate_hashes:
        raise ValueError(
            "Baseline and candidate datasets differ"
        )

    regressions = []
    comparisons = []
    category_summary = defaultdict(list)

    for test_id in sorted(baseline):
        old = baseline[test_id]
        new = candidate[test_id]

        if old["category"] != new["category"]:
            raise ValueError(
                f"Category mismatch for {test_id}"
            )

        for metric in METRICS:
            old_score = old[metric]
            new_score = new[metric]
            delta = new_score - old_score

            comparison = {
                "test_id": test_id,
                "category": old["category"],
                "metric": metric,
                "baseline": old_score,
                "candidate": new_score,
                "delta": delta,
            }

            comparisons.append(comparison)
            category_summary[old["category"]].append(comparison)

            if delta < -tolerance - 1e-9:
                regressions.append(comparison)

    return {
        "passed": len(regressions) == 0,
        "tolerance": tolerance,
        "test_count": len(baseline),
        "regressions": regressions,
        "comparisons": comparisons,
        "category_summary": {
            category: {
                "regression_count": sum(
                    1
                    for comparison in values
                    if comparison in regressions
                ),
                "comparison_count": len(values),
            }
            for category, values in category_summary.items()
        },
    }
