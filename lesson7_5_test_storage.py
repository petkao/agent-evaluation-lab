
from evaluation.models import create_evaluation_result
from evaluation.storage import (
    save_evaluation_result,
    load_evaluation_results,
)


def main():

    scores = {
        "tool_call_accuracy": 1.0,
        "tool_call_f1": 1.0
    }

    result = create_evaluation_result(
        test_id="hotel_001",
        agent_version="v1.0",
        scores=scores
    )

    # Save result to JSONL
    save_evaluation_result(result)

    # Load all saved results
    saved_results = load_evaluation_results()

    print("\nStored Evaluation Results")
    print("-------------------------")

    for item in saved_results:
        print(item)


if __name__ == "__main__":
    main()

# poetry run python lesson7_5_test_storage.py
