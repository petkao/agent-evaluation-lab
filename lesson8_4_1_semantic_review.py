
import asyncio
import json
from pathlib import Path

from evaluation.semantic_reviewer import review_candidate
from evaluation.provenance import candidate_sha256


INPUT_PATH = Path(
    "datasets/hotel_synthetic_candidates_v1.jsonl"
)

OUTPUT_PATH = Path(
    "results/semantic_reviews_v1.jsonl"
)


def load_candidates(path: Path):

    candidates = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                candidates.append(json.loads(line))

    return candidates


async def main():

    candidates = load_candidates(INPUT_PATH)

    print("\nLesson 8.4.1 — Semantic Review")
    print("------------------------------")
    print("Candidates:", len(candidates))

    reviews = []

    for candidate in candidates:

        review = await review_candidate(candidate)

        # Preserve the original dataset ID as the authoritative ID.
        review_record = review.model_dump()
        review_record["test_id"] = candidate["test_id"]
        review_record["candidate_sha256"] = candidate_sha256(
            candidate
        )

        reviews.append(review_record)

        status = (
            "PASS"
            if review.is_valid
            else "REVIEW REQUIRED"
        )

        print(
            f"{candidate['test_id']}: "
            f"{status} "
            f"(confidence={review.confidence:.2f})"
        )

        print("  Reason:", review.reason)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Overwrite the report for this run.
    # Do not append duplicate reviews on repeated runs.
    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        for review in reviews:
            file.write(
                json.dumps(review) + "\n"
            )

    passed = sum(
        review["is_valid"]
        for review in reviews
    )

    print("\nReview Summary")
    print("--------------")
    print("Passed:", passed)
    print("Flagged:", len(reviews) - passed)
    print("Report:", OUTPUT_PATH)


if __name__ == "__main__":
    asyncio.run(main())

# poetry run python lesson8_4_1_semantic_review.py