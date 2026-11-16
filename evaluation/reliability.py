
from collections import defaultdict
from statistics import mean, stdev


METRICS = (
    "tool_call_accuracy",
    "tool_call_f1",
)


def analyze_reliability(records):
    if not records:
        raise ValueError("No evaluation records provided")

    run_ids = {r["run_id"] for r in records}
    dataset_hashes = {r["dataset_sha256"] for r in records}
    agent_versions = {r["agent_version"] for r in records}

    if len(dataset_hashes) != 1:
        raise ValueError("Multiple golden datasets detected")

    if len(agent_versions) != 1:
        raise ValueError("Multiple agent versions detected")

    grouped = defaultdict(list)
    run_test_pairs = set()

    for record in records:
        key = (record["run_id"], record["test_id"])

        if key in run_test_pairs:
            raise ValueError(
                f"Duplicate evaluation record: {key}"
            )

        run_test_pairs.add(key)
        grouped[record["test_id"]].append(record)

    expected_test_ids = set(grouped)

    # Every run must contain the same test cases.
    for run_id in run_ids:
        actual_ids = {
            r["test_id"]
            for r in records
            if r["run_id"] == run_id
        }

        if actual_ids != expected_test_ids:
            raise ValueError(
                f"Incomplete evaluation run: {run_id}"
            )

    per_test = {}

    for test_id, evaluations in sorted(grouped.items()):
        category = evaluations[0]["category"]

        if any(r["category"] != category for r in evaluations):
            raise ValueError(
                f"Inconsistent category: {test_id}"
            )

        scores = {
            metric: [r[metric] for r in evaluations]
            for metric in METRICS
        }

        passes = sum(
            all(
                r[metric] >= 1.0 - 1e-9
                for metric in METRICS
            )
            for r in evaluations
        )

        per_test[test_id] = {
            "category": category,
            "runs": len(evaluations),
            "passes": passes,
            "pass_rate": passes / len(evaluations),
            "metrics": {
                metric: {
                    "mean": mean(values),
                    "min": min(values),
                    "max": max(values),
                    "stdev": (
                        stdev(values)
                        if len(values) > 1
                        else 0.0
                    ),
                }
                for metric, values in scores.items()
            },
        }

    category_tests = defaultdict(list)

    for test_id, summary in per_test.items():
        category_tests[summary["category"]].append(summary)

    category_summary = {}

    for category, tests in category_tests.items():
        category_summary[category] = {
            "test_count": len(tests),
            "mean_pass_rate": mean(
                test["pass_rate"] for test in tests
            ),
            "minimum_pass_rate": min(
                test["pass_rate"] for test in tests
            ),
        }

    return {
        "agent_version": next(iter(agent_versions)),
        "dataset_sha256": next(iter(dataset_hashes)),
        "run_count": len(run_ids),
        "test_count": len(per_test),
        "overall_mean_pass_rate": mean(
            test["pass_rate"] for test in per_test.values()
        ),
        "per_test": per_test,
        "categories": category_summary,
    }
