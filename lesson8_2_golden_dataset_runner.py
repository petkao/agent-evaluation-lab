
import asyncio
from statistics import mean

from langchain_core.messages import HumanMessage

from lesson5_2_langgraph_trajectory import graph
from evaluation.dataset_loader import load_golden_dataset
from evaluation.tool_evaluator import evaluate_tool_calls
from evaluation.models import create_evaluation_result
from evaluation.storage import save_evaluation_result


DATASET_PATH = "datasets/hotel_golden_v1.jsonl"
RESULTS_PATH = "results/golden_evaluations_v1.jsonl"
AGENT_VERSION = "v1.0"


async def main():

    # 1. Load the golden dataset
    test_cases = load_golden_dataset(DATASET_PATH)

    print("\nGolden Dataset Evaluation")
    print("-------------------------")
    print("Total Test Cases:", len(test_cases))

    all_scores = []

    # 2. Evaluate each test case
    for test_case in test_cases:

        test_id = test_case["test_id"]

        print(f"\nRunning Test: {test_id}")

        # Execute the real LangGraph agent
        result = await graph.ainvoke({
            "messages": [
                HumanMessage(
                    content=test_case["user_input"]
                )
            ]
        })

        # Evaluate tool calls
        scores = await evaluate_tool_calls(
            langgraph_messages=result["messages"],
            expected_calls=test_case["expected_tool_calls"]
        )

        # Create structured result
        evaluation_result = create_evaluation_result(
            test_id=test_id,
            agent_version=AGENT_VERSION,
            scores=scores
        )

        # Persist the result
        save_evaluation_result(
            evaluation_result,
            file_path=RESULTS_PATH
        )

        all_scores.append(scores)

        print(
            f"  Tool Accuracy: {scores['tool_call_accuracy']:.3f}"
        )
        print(
            f"  Tool F1:       {scores['tool_call_f1']:.3f}"
        )

    # 3. Calculate aggregate metrics
    if all_scores:

        avg_accuracy = mean(
            score["tool_call_accuracy"]
            for score in all_scores
        )

        avg_f1 = mean(
            score["tool_call_f1"]
            for score in all_scores
        )

        print("\nAggregate Evaluation Results")
        print("----------------------------")
        print(f"Average Tool Accuracy: {avg_accuracy:.3f}")
        print(f"Average Tool F1:       {avg_f1:.3f}")

    print("\nResults saved to:", RESULTS_PATH)


if __name__ == "__main__":
    asyncio.run(main())

# poetry run python lesson8_2_golden_dataset_runner.py