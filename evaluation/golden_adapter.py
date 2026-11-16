
import json
from pathlib import Path

from evaluation.golden_schema import GoldenTestCase


def infer_category(tool_calls: list[dict]) -> str:
    names = {call["name"] for call in tool_calls}

    if names == {"search_hotels"}:
        return "hotel_only"

    if names == {"get_weather"}:
        return "weather_only"

    if names == {"search_hotels", "get_weather"}:
        return "combined"

    return "other"


def convert_legacy_case(
    legacy_case: dict,
    agent_id: str = "travel_agent",
) -> GoldenTestCase:
    tool_calls = legacy_case["expected_tool_calls"]

    converted = {
        "schema_version": "1.0",
        "dataset_version": legacy_case.get("dataset_version"),
        "test_id": legacy_case["test_id"],
        "category": infer_category(tool_calls),
        "agent_id": agent_id,
        "user_input": legacy_case["user_input"],
        "expected": {
            "required_tool_calls": tool_calls
        }
    }

    return GoldenTestCase.model_validate(converted)


def load_legacy_golden_dataset(
    file_path: str,
    agent_id: str = "travel_agent",
) -> list[GoldenTestCase]:
    test_cases = []

    with Path(file_path).open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            legacy_case = json.loads(line)

            try:
                golden = convert_legacy_case(
                    legacy_case,
                    agent_id=agent_id,
                )
            except Exception as exc:
                raise ValueError(
                    f"Invalid golden test case at line {line_number}"
                ) from exc

            test_cases.append(golden)

    return test_cases
