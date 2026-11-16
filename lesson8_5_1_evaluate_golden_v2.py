
import asyncio
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.messages import HumanMessage

from evaluation.dataset_loader import load_golden_dataset
from evaluation.tool_evaluator import evaluate_tool_calls
from evaluation.batch_approval import sha256_file
from lesson5_2_langgraph_trajectory import graph


DATASET_PATH = "datasets/hotel_golden_v2.jsonl"
OUTPUT_PATH = "results/golden_evaluations_v2.jsonl"
SUMMARY_PATH = "results/golden_summary_v2.json"

AGENT_VERSION = "travel_agent_baseline_v1"


def classify_case(expected_calls):
    names = {call.name for call in expected_calls}

    if names == {"search_hotels"}:
        return "hotel_only"

    if names == {"get_weather"}:
        return "weather_only"

    if names == {"search_hotels", "get_weather"}:
        return "combined"

    raise ValueError(f"Unexpected tool combination: {names}")


async def main():
    dataset = load_golden_dataset(DATASET_PATH)

    if len(dataset) != 10:
        raise ValueError(
            f"Expected 10 golden cases, found {len(dataset)}"
        )

    dataset_hash = sha256_file(DATASET_PATH)
    run_id = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )

    results = []
    category_scores = defaultdict(list)

    print("Lesson 8.5.1 — Golden Dataset v2 Evaluation")
    print("-------------------------------------------")
    print("Agent version:", AGENT_VERSION)
    print("Dataset version: v2.0")
    print("Test cases:", len(dataset))
    print()

    for case in dataset:
        test_id = case["test_id"]
        user_input = case["user_input"]
        expected_calls = case["expected_tool_calls"]

        category = classify_case(expected_calls)

        print(f"Running {test_id} ({category})...")

        result = await graph.ainvoke({
            "messages": [
                HumanMessage(content=user_input)
            ]
        })

        scores = await evaluate_tool_calls(
            result["messages"],
            expected_calls
        )

        record = {
            "run_id": run_id,
            "test_id": test_id,
            "agent_version": AGENT_VERSION,
            "dataset_version": "v2.0",
            "dataset_sha256": dataset_hash,
            "category": category,
            "tool_call_accuracy": scores["tool_call_accuracy"],
            "tool_call_f1": scores["tool_call_f1"],
            "evaluated_at": datetime.now(
                timezone.utc
            ).isoformat()
        }

        results.append(record)
        category_scores[category].append(record)

        print(
            f"  Accuracy: {record['tool_call_accuracy']:.3f}"
        )
        print(
            f"  F1:       {record['tool_call_f1']:.3f}"
        )

    output_path = Path(OUTPUT_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Preserve each evaluation run rather than overwriting history.
    with output_path.open("a", encoding="utf-8") as file:
        for record in results:
            file.write(json.dumps(record) + "\n")

    def summarize(records):
        return {
            "count": len(records),
            "mean_accuracy": sum(
                r["tool_call_accuracy"] for r in records
            ) / len(records),
            "mean_f1": sum(
                r["tool_call_f1"] for r in records
            ) / len(records)
        }

    summary = {
        "run_id": run_id,
        "agent_version": AGENT_VERSION,
        "dataset_version": "v2.0",
        "dataset_sha256": dataset_hash,
        "overall": summarize(results),
        "categories": {
            category: summarize(records)
            for category, records in category_scores.items()
        }
    }

    summary_path = Path(SUMMARY_PATH)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8"
    )

    print("\nEvaluation Summary")
    print("------------------")

    print(
        "Overall accuracy:",
        f"{summary['overall']['mean_accuracy']:.3f}"
    )
    print(
        "Overall F1:",
        f"{summary['overall']['mean_f1']:.3f}"
    )

    print("\nCategory Results")

    for category, metrics in summary["categories"].items():
        print(
            f"{category}: "
            f"count={metrics['count']}, "
            f"accuracy={metrics['mean_accuracy']:.3f}, "
            f"F1={metrics['mean_f1']:.3f}"
        )

    print("\nResults saved:", OUTPUT_PATH)
    print("Summary saved:", SUMMARY_PATH)


if __name__ == "__main__":
    asyncio.run(main())


# poetry run python lesson8_5_1_evaluate_golden_v2.py
