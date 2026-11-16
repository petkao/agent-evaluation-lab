
import json
import sys
from collections import defaultdict
from pathlib import Path

from evaluation.regression_gate import compare_evaluation_runs


RESULTS_PATH = Path("results/golden_evaluations_v2.jsonl")


def load_runs(path):
    runs = defaultdict(list)

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            record = json.loads(line)
            runs[record["run_id"]].append(record)

    return runs


def main():
    runs = load_runs(RESULTS_PATH)

    if len(runs) < 2:
        raise ValueError(
            "At least two evaluation runs are required"
        )

    # Python dictionaries preserve insertion order.
    run_ids = list(runs)

    baseline_id = run_ids[0]
    candidate_id = run_ids[-1]

    baseline = runs[baseline_id]
    candidate = runs[candidate_id]

    print("Lesson 8.5.3 — Actual Run Comparison")
    print("------------------------------------")
    print("Baseline run:", baseline_id)
    print("Candidate run:", candidate_id)
    print("Baseline cases:", len(baseline))
    print("Candidate cases:", len(candidate))

    # We are comparing repeated executions of the
    # same agent, not two different agent versions.
    baseline_versions = {
        r["agent_version"] for r in baseline
    }
    candidate_versions = {
        r["agent_version"] for r in candidate
    }

    if len(baseline_versions) != 1:
        raise ValueError("Mixed baseline agent versions")

    if len(candidate_versions) != 1:
        raise ValueError("Mixed candidate agent versions")

    print("Baseline agent:", next(iter(baseline_versions)))
    print("Candidate agent:", next(iter(candidate_versions)))

    report = compare_evaluation_runs(
        baseline,
        candidate,
        tolerance=0.0
    )

    print("\nRegression Report")
    print("-----------------")
    print("Compared cases:", report["test_count"])
    print("Regressed metrics:", len(report["regressions"]))
    print("Status:", "PASS" if report["passed"] else "FAIL")

    for regression in report["regressions"]:
        print(
            f"{regression['test_id']} "
            f"[{regression['metric']}]: "
            f"{regression['baseline']:.3f} -> "
            f"{regression['candidate']:.3f}"
        )

    if not report["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()

# poetry run python lesson8_5_3_compare_runs.py