
from evaluation.dataset_loader import load_golden_dataset


def main():

    dataset = load_golden_dataset(
        "datasets/hotel_golden_v1.jsonl"
    )

    print("\nGolden Dataset")
    print("--------------")

    print("Total Test Cases:", len(dataset))

    for test_case in dataset:

        print("\nTest ID:", test_case["test_id"])

        print(
            "Dataset Version:",
            test_case["dataset_version"]
        )

        print(
            "User Input:",
            test_case["user_input"]
        )

        print("Expected Tool Calls:")

        for call in test_case["expected_tool_calls"]:
            print(" ", call.name, call.args)


if __name__ == "__main__":
    main()

# poetry run python lesson8_1_test_golden_dataset.py
