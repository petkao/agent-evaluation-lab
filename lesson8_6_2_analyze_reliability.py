
import json
from pathlib import Path

from evaluation.reliability import analyze_reliability


RESULTS_PATH = Path("results/golden_evaluations_v2.jsonl")
OUTPUT_PATH = Path("results/reliability_summary_v2.json")


def main():
    records = []

    with RESULTS_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))

    summary = analyze_reliability(records)

    print("Lesson 8.6 — Agent Reliability Analysis")
    print("---------------------------------------")
    print("Agent:", summary["agent_version"])
    print("Evaluation runs:", summary["run_count"])
    print("Golden test cases:", summary["test_count"])

    print(
        "Overall mean pass rate:",
        f"{summary['overall_mean_pass_rate']:.1%}"
    )

    print("\nCategory Reliability")
    print("--------------------")

    for category, metrics in summary["categories"].items():
        print(
            f"{category}: "
            f"tests={metrics['test_count']}, "
            f"mean_pass_rate={metrics['mean_pass_rate']:.1%}, "
            f"minimum_pass_rate={metrics['minimum_pass_rate']:.1%}"
        )

    print("\nPer-Test Reliability")
    print("--------------------")

    for test_id, metrics in summary["per_test"].items():
        print(
            f"{test_id}: "
            f"{metrics['passes']}/{metrics['runs']} passes, "
            f"pass_rate={metrics['pass_rate']:.1%}"
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8"
    )

    print("\nSummary saved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()

# poetry run python lesson8_6_2_analyze_reliability.py
