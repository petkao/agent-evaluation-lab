
import json
from datetime import datetime, timezone
from pathlib import Path

from evaluation.batch_approval import (
    load_jsonl,
    sha256_file
)
from evaluation.provenance import candidate_sha256


def promote_golden_dataset(
    candidate_path,
    review_path,
    manifest_path,
    output_path,
    audit_path,
    approver,
    approval_reference
):
    candidate_path = Path(candidate_path)
    review_path = Path(review_path)
    manifest_path = Path(manifest_path)
    output_path = Path(output_path)
    audit_path = Path(audit_path)

    if not approver.strip():
        raise ValueError("Approver is required")

    if not approval_reference.strip():
        raise ValueError("Approval reference is required")

    if output_path.exists() or audit_path.exists():
        raise FileExistsError(
            "Golden dataset or audit record already exists"
        )

    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )

    if manifest["status"] != "pending_human_review":
        raise ValueError("Unexpected manifest status")

    if manifest.get("approval_policy") != "human_batch_review_v1":
        raise ValueError("Unsupported approval policy")

    # Verify that the source files have not changed.
    if sha256_file(candidate_path) != manifest["candidate_sha256"]:
        raise ValueError("Candidate dataset changed")

    if sha256_file(review_path) != manifest["review_sha256"]:
        raise ValueError("Semantic review report changed")

    candidates = load_jsonl(candidate_path)
    reviews = load_jsonl(review_path)

    candidate_ids = [c["test_id"] for c in candidates]
    review_ids = [r["test_id"] for r in reviews]

    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("Duplicate candidate IDs")

    if len(review_ids) != len(set(review_ids)):
        raise ValueError("Duplicate review IDs")

    if set(candidate_ids) != set(review_ids):
        raise ValueError("Candidate/review ID mismatch")

    if len(candidates) != manifest["total_candidates"]:
        raise ValueError("Candidate count mismatch")

    review_by_id = {
        r["test_id"]: r for r in reviews
    }

    passed_ids = set()
    flagged_ids = set()

    for candidate in candidates:
        test_id = candidate["test_id"]
        review = review_by_id[test_id]

        if review.get("candidate_sha256") != candidate_sha256(candidate):
            raise ValueError(
                f"Provenance mismatch: {test_id}"
            )

        if review.get("is_valid") is True:
            passed_ids.add(test_id)
        elif review.get("is_valid") is False:
            flagged_ids.add(test_id)
        else:
            raise ValueError(
                f"Invalid semantic decision: {test_id}"
            )

    if len(passed_ids) != manifest["passed_count"]:
        raise ValueError("Passed count mismatch")

    if len(flagged_ids) != manifest["flagged_count"]:
        raise ValueError("Flagged count mismatch")

    if flagged_ids != set(manifest["flagged_test_ids"]):
        raise ValueError("Flagged IDs changed")

    sampled_ids = manifest["sampled_test_ids"]

    if len(sampled_ids) != len(set(sampled_ids)):
        raise ValueError("Duplicate sampled IDs")

    if len(sampled_ids) != manifest["sample_size"]:
        raise ValueError("Sample size mismatch")

    if not set(sampled_ids).issubset(passed_ids):
        raise ValueError("Sample contains invalid IDs")

    # Current approval policy requires all 10
    # candidates to be included in the reviewed batch.
    if flagged_ids:
        raise ValueError(
            "Flagged candidates require resolution"
        )

    if set(sampled_ids) != passed_ids:
        raise ValueError(
            "All passing candidates must be sampled"
        )

    if len(candidates) != 10:
        raise ValueError(
            "This approval covers exactly 10 candidates"
        )

    golden_records = []

    for candidate in candidates:
        golden_records.append({
            "test_id": candidate["test_id"],
            "dataset_version": "v2.0",
            "user_input": candidate["user_input"],
            "expected_tool_calls": candidate["expected_tool_calls"]
        })

    output_path.parent.mkdir(
        parents=True, exist_ok=True
    )
    audit_path.parent.mkdir(
        parents=True, exist_ok=True
    )

    # Exclusive creation prevents accidental overwrite.
    with output_path.open("x", encoding="utf-8") as file:
        for record in golden_records:
            file.write(json.dumps(record) + "\n")

    audit_record = {
        "dataset_version": "v2.0",
        "status": "approved",
        "approver": approver,
        "approval_reference": approval_reference,
        "approved_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "approval_policy": manifest["approval_policy"],
        "candidate_sha256": manifest["candidate_sha256"],
        "review_sha256": manifest["review_sha256"],
        "manifest_sha256": sha256_file(manifest_path),
        "golden_sha256": sha256_file(output_path),
        "approved_count": len(golden_records),
        "approved_test_ids": candidate_ids
    }

    with audit_path.open("x", encoding="utf-8") as file:
        json.dump(audit_record, file, indent=2)

    return audit_record
