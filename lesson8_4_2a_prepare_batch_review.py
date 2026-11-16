
import json
from pathlib import Path

from evaluation.batch_approval import prepare_batch_review


CANDIDATE_PATH = Path(
    "datasets/hotel_synthetic_candidates_v1.jsonl"
)

REVIEW_PATH = Path(
    "results/semantic_reviews_v1.jsonl"
)

MANIFEST_PATH = Path(
    "results/batch_review_manifest_v1.json"
)


def main():

    manifest, sampled, flagged = prepare_batch_review(
        candidate_path=CANDIDATE_PATH,
        review_path=REVIEW_PATH,
        sample_size=10,
        seed=42
    )

    MANIFEST_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with MANIFEST_PATH.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)

    print("\nLesson 8.4.2A — Batch Review Preparation")
    print("----------------------------------------")

    print("Total Candidates:", manifest["total_candidates"])
    print("Passed:", manifest["passed_count"])
    print("Flagged:", manifest["flagged_count"])
    print("Sampled:", manifest["sample_size"])

    print("\nSampled Candidates")

    for candidate in sampled:
        print(
            f"\n{candidate['test_id']}: "
            f"{candidate['user_input']}"
        )

        for call in candidate["expected_tool_calls"]:
            print(
                f"  {call['name']}: {call['args']}"
            )

    print("\nFlagged Candidates")

    for candidate in flagged:
        print(
            f"{candidate['test_id']}: "
            f"{candidate['user_input']}"
        )

    print("\nManifest saved to:", MANIFEST_PATH)


if __name__ == "__main__":
    main()

# poetry run python lesson8_4_2a_prepare_batch_review.py
