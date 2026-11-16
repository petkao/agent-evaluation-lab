
from evaluation.golden_adapter import load_legacy_golden_dataset


def main():
    print("Lesson 9.1 — Legacy Golden Dataset Adapter")
    print("-" * 50)

    dataset_path = "datasets/hotel_golden_v2.jsonl"

    golden_cases = load_legacy_golden_dataset(dataset_path)

    for case in golden_cases:
        assert case.schema_version == "1.0"
        assert case.dataset_version == "v2.0"

    print("Schema version: 1.0")
    print("Dataset version: v2.0")
    print("Version preservation: PASS")

    print("Loaded test cases:", len(golden_cases))

    assert len(golden_cases) == 10, (
        f"Expected 10 golden cases, got {len(golden_cases)}"
    )

    categories = {}

    for case in golden_cases:
        categories[case.category] = categories.get(case.category, 0) + 1

        print(
            f"{case.test_id}: "
            f"category={case.category}, "
            f"tools={len(case.expected.required_tool_calls)}"
        )

    print("\nCategory coverage:")

    for category, count in sorted(categories.items()):
        print(f"  {category}: {count}")

    assert categories == {
        "hotel_only": 4,
        "weather_only": 3,
        "combined": 3,
    }, f"Unexpected category distribution: {categories}"

    print("\nLegacy dataset conversion: PASS")
    print("All Lesson 9.1 adapter tests PASSED")


if __name__ == "__main__":
    main()  

# poetry run python lesson9_1_test_legacy_adapter.py
