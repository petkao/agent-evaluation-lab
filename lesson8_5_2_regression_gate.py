
import copy
import json
import sys
from pathlib import Path

from evaluation.regression_gate import compare_evaluation_runs


RESULTS_PATH = Path("results/golden_evaluations_v2.jsonl")


def load_latest_run(file_path):
    records = [
        json.loads(line)
        for line in file_path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    if not records:
        raise ValueError("No evaluation records found")

    latest_run_id = records[-1]["run_id"]

    return [
        record
        for record in records
        if record["run_id"] == latest_run_id
    ]


def print_report(report):
    print("\nRegression Gate Report")
    print("----------------------")
    print("Test cases:", report["test_count"])
    print("Tolerance:", report["tolerance"])
    print("Regressions:", len(report["regressions"]))

    if report["passed"]:
        print("Status: PASS")
    else:
        print("Status: FAIL")

        for regression in report["regressions"]:
            print(
                f"  {regression['test_id']} "
                f"[{regression['metric']}]: "
                f"{regression['baseline']:.3f} -> "
                f"{regression['candidate']:.3f}"
            )


def main():
    baseline = load_latest_run(RESULTS_PATH)

    if len(baseline) != 10:
        raise ValueError(
            f"Expected 10 baseline cases, found {len(baseline)}"
        )

    print("Lesson 8.5.2 — Regression Gate")
    print("==============================")

    # Test A: identical candidate should pass.
    candidate = copy.deepcopy(baseline)

    report = compare_evaluation_runs(
        baseline,
        candidate,
        tolerance=0.0
    )

    print("\nTEST A — Identical Candidate")
    print_report(report)

    if not report["passed"]:
        sys.exit(1)

    # Test B: deliberately introduce a regression.
    candidate = copy.deepcopy(baseline)

    target = next(
        r for r in candidate
        if r["test_id"] == "synthetic_008"
    )

    target["tool_call_accuracy"] = 0.0
    target["tool_call_f1"] = 0.5

    report = compare_evaluation_runs(
        baseline,
        candidate,
        tolerance=0.0
    )

    print("\nTEST B — Simulated Regression")
    print_report(report)

    # A regression is expected in this test.
    if report["passed"]:
        print("ERROR: Regression gate failed to detect regression")
        sys.exit(1)

    print("\nRegression gate self-test PASSED")


if __name__ == "__main__":
    main()


# poetry run python lesson8_5_2_regression_gate.py
