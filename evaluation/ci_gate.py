
import json
from pathlib import Path

from evaluation.batch_approval import sha256_file


METRICS = ("tool_call_accuracy", "tool_call_f1")


def validate_ci_gate(
    dataset_path,
    audit_path,
    evaluation_path,
    minimum_score=1.0,
):
    dataset_path = Path(dataset_path)
    audit_path = Path(audit_path)
    evaluation_path = Path(evaluation_path)

    audit = json.loads(
        audit_path.read_text(encoding="utf-8")
    )

    if audit.get("status") != "approved":
        raise ValueError("Golden dataset is not approved")

    actual_hash = sha256_file(dataset_path)
    approved_hash = audit["golden_sha256"]

    if actual_hash != approved_hash:
        raise ValueError("Golden dataset integrity failure")

    dataset_records = [
        json.loads(line)
        for line in dataset_path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    expected_ids = [r["test_id"] for r in dataset_records]

    if len(expected_ids) != len(set(expected_ids)):
        raise ValueError("Duplicate golden test IDs")

    evaluation_records = [
        json.loads(line)
        for line in evaluation_path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    if not evaluation_records:
        raise ValueError("No evaluation records found")

    latest_run_id = evaluation_records[-1]["run_id"]

    latest_run = [
        r for r in evaluation_records
        if r["run_id"] == latest_run_id
    ]

    actual_ids = [r["test_id"] for r in latest_run]

    if len(actual_ids) != len(set(actual_ids)):
        raise ValueError("Duplicate evaluation test IDs")

    if set(actual_ids) != set(expected_ids):
        raise ValueError("Incomplete evaluation run")

    for record in latest_run:
        if record["dataset_sha256"] != approved_hash:
            raise ValueError(
                f"Dataset hash mismatch: {record['test_id']}"
            )

        for metric in METRICS:
            if record[metric] < minimum_score:
                raise ValueError(
                    f"Quality gate failed: "
                    f"{record['test_id']} "
                    f"{metric}={record[metric]:.3f}"
                )

    return {
        "passed": True,
        "run_id": latest_run_id,
        "test_count": len(latest_run),
        "minimum_score": minimum_score,
    }
