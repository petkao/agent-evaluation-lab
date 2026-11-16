
import hashlib
import json
import random
from pathlib import Path


def load_jsonl(file_path):
    records = []

    with Path(file_path).open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))

    return records


def sha256_file(file_path):
    digest = hashlib.sha256()

    with Path(file_path).open("rb") as file:
        for chunk in iter(lambda: file.read(65536), b""):
            digest.update(chunk)

    return digest.hexdigest()


def prepare_batch_review(
    candidate_path,
    review_path,
    sample_size=10,
    seed=42
):
    candidates = load_jsonl(candidate_path)
    reviews = load_jsonl(review_path)

    candidate_ids = [c["test_id"] for c in candidates]
    review_ids = [r["test_id"] for r in reviews]

    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("Duplicate candidate test IDs")

    if len(review_ids) != len(set(review_ids)):
        raise ValueError("Duplicate semantic review test IDs")

    if set(candidate_ids) != set(review_ids):
        raise ValueError(
            "Candidate IDs and review IDs do not match"
        )

    review_by_id = {
        review["test_id"]: review
        for review in reviews
    }

    passed = []
    flagged = []

    for candidate in candidates:
        review = review_by_id[candidate["test_id"]]

        if review["is_valid"] is True:
            passed.append(candidate)
        else:
            flagged.append(candidate)

    rng = random.Random(seed)

    selected = rng.sample(
        passed,
        k=min(sample_size, len(passed))
    )

    manifest = {
        "status": "pending_human_review",
        "approval_policy": "human_batch_review_v1",
        "candidate_file": str(candidate_path),
        "review_file": str(review_path),
        "candidate_sha256": sha256_file(candidate_path),
        "review_sha256": sha256_file(review_path),
        "total_candidates": len(candidates),
        "passed_count": len(passed),
        "flagged_count": len(flagged),
        "sample_seed": seed,
        "sample_size": len(selected),
        "sampled_test_ids": [
            candidate["test_id"]
            for candidate in selected
        ],
        "flagged_test_ids": [
            candidate["test_id"]
            for candidate in flagged
        ]
    }

    return manifest, selected, flagged
