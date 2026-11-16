
import json
from pathlib import Path

from ragas.messages import ToolCall


def load_golden_dataset(file_path: str):

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Golden dataset not found: {file_path}"
        )

    test_cases = []

    with path.open("r", encoding="utf-8") as file:

        for line_number, line in enumerate(file, start=1):

            if not line.strip():
                continue

            data = json.loads(line)

            expected_calls = [
                ToolCall(
                    name=call["name"],
                    args=call["args"]
                )
                for call in data["expected_tool_calls"]
            ]

            test_cases.append({
                "test_id": data["test_id"],
                "dataset_version": data["dataset_version"],
                "user_input": data["user_input"],
                "expected_tool_calls": expected_calls
            })

    return test_cases
