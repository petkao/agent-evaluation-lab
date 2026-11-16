
import sys

from evaluation.ci_gate import validate_ci_gate


def main():
    try:
        report = validate_ci_gate(
            dataset_path="datasets/hotel_golden_v2.jsonl",
            audit_path="datasets/hotel_golden_v2_approval.json",
            evaluation_path="results/golden_evaluations_v2.jsonl",
            minimum_score=1.0,
        )

        print("Lesson 8.7 — CI Quality Gate")
        print("----------------------------")
        print("Run:", report["run_id"])
        print("Test cases:", report["test_count"])
        print("Minimum score:", report["minimum_score"])
        print("Status: PASS")

    except (ValueError, KeyError, OSError) as error:
        print("Lesson 8.7 — CI Quality Gate")
        print("----------------------------")
        print("Status: FAIL")
        print("Reason:", error)
        sys.exit(1)


if __name__ == "__main__":
    main()

# poetry run python lesson8_7_1_ci_gate.py
