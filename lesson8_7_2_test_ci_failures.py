
import copy
import json
import tempfile
from pathlib import Path

from evaluation.ci_gate import validate_ci_gate


DATASET_PATH = "datasets/hotel_golden_v2.jsonl"
AUDIT_PATH = "datasets/hotel_golden_v2_approval.json"
EVALUATION_PATH = Path(
    "results/golden_evaluations_v2.jsonl"
)


def load_records(path):
    return [
        json.loads(line)
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]


def save_records(path, records):
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record) + "\n")


def expect_failure(name, records, temp_dir):
    test_path = temp_dir / "test_evaluations.jsonl"
    save_records(test_path, records)

    try:
        validate_ci_gate(
            dataset_path=DATASET_PATH,
            audit_path=AUDIT_PATH,
            evaluation_path=test_path,
            minimum_score=1.0,
        )
    except (ValueError, KeyError, OSError) as error:
        print(f"{name}: PASS")
        print(f"  Gate rejected: {error}")
        return

    raise AssertionError(
        f"{name}: Gate incorrectly accepted bad results"
    )


def main():
    records = load_records(EVALUATION_PATH)

    latest_run_id = records[-1]["run_id"]

    latest = [
        record
        for record in records
        if record["run_id"] == latest_run_id
    ]

    assert len(latest) == 10

    print("Lesson 8.7.2 — CI Failure Tests")
    print("--------------------------------")

    with tempfile.TemporaryDirectory() as directory:
        temp_dir = Path(directory)

        # Test A: Lower the accuracy of one test.
        degraded = copy.deepcopy(latest)
        degraded[0]["tool_call_accuracy"] = 0.5

        expect_failure(
            "TEST A — Degraded score",
            degraded,
            temp_dir,
        )

        # Test B: Remove one evaluation case.
        incomplete = copy.deepcopy(latest[:-1])

        expect_failure(
            "TEST B — Incomplete run",
            incomplete,
            temp_dir,
        )

        # Test C: Alter the recorded dataset hash.
        wrong_hash = copy.deepcopy(latest)
        wrong_hash[0]["dataset_sha256"] = "invalid_hash"

        expect_failure(
            "TEST C — Dataset hash mismatch",
            wrong_hash,
            temp_dir,
        )

    print("\nAll CI failure tests PASSED")


if __name__ == "__main__":
    main()

# poetry run python lesson8_7_2_test_ci_failures.py
