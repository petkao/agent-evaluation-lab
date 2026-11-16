
import json
from pathlib import Path

from evaluation.golden_promoter import promote_golden_dataset


def main():

    manifest_path = Path(
        "results/batch_review_manifest_v1.json"
    )

    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )

    print("\nLesson 8.4.2B — Golden Dataset Promotion")
    print("----------------------------------------")
    print("Candidates:", manifest["total_candidates"])
    print("Passed:", manifest["passed_count"])
    print("Flagged:", manifest["flagged_count"])
    print("Sampled:", manifest["sample_size"])

    print("\nApproval is for the refreshed review manifest.")
    print("Please verify its contents before confirming.")

    confirmation = input(
        "\nType APPROVE V2 to promote this batch: "
    ).strip()

    if confirmation != "APPROVE V2":
        print("Promotion cancelled.")
        return

    audit = promote_golden_dataset(
        candidate_path="datasets/hotel_synthetic_candidates_v1.jsonl",
        review_path="results/semantic_reviews_v1.jsonl",
        manifest_path=manifest_path,
        output_path="datasets/hotel_golden_v2.jsonl",
        audit_path="datasets/hotel_golden_v2_approval.json",
        approver="dataset_owner",
        approval_reference="interactive_batch_approval_v2"
    )

    print("\nGolden Dataset v2 Created")
    print("-------------------------")
    print("Approved Records:", audit["approved_count"])
    print("Dataset:", "datasets/hotel_golden_v2.jsonl")
    print("Audit:", "datasets/hotel_golden_v2_approval.json")
    print("SHA-256:", audit["golden_sha256"])


if __name__ == "__main__":
    main()

# poetry run python lesson8_4_2b_promote_golden.py
