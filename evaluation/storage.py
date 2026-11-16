
import json
from pathlib import Path

from evaluation.models import EvaluationResult


def save_evaluation_result(
    result: EvaluationResult,
    file_path: str = "results/evaluations.jsonl"
):
    """
    Append one evaluation result to a JSONL file.
    """

    path = Path(file_path)

    # Create the directory if it doesn't exist.
    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Append the result as one JSON line.
    with path.open("a", encoding="utf-8") as file:
        file.write(
            json.dumps(result.to_dict()) + "\n"
        )


def load_evaluation_results(
    file_path: str = "results/evaluations.jsonl"
):
    """
    Load evaluation results from a JSONL file.
    """

    path = Path(file_path)

    if not path.exists():
        return []

    results = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                results.append(json.loads(line))

    return results
