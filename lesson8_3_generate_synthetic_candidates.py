
from dotenv import load_dotenv

from evaluation.candidate_validator import validate_dataset_coverage

load_dotenv()

import asyncio
import json
from pathlib import Path

from evaluation.synthetic_generator import (
    generate_synthetic_candidates
)
from evaluation.candidate_validator import (
    validate_batch
)


OUTPUT_PATH = Path(
    "datasets/hotel_synthetic_candidates_v1.jsonl"
)


async def main():

    print("\nGenerating Synthetic Test Cases")
    print("-------------------------------")

    candidates = await generate_synthetic_candidates()

    print("Generated Candidates:", len(candidates))

    valid, rejected = validate_batch(candidates)

    counts, coverage_errors = validate_dataset_coverage(valid)

    print("\nDataset Coverage:", counts)

    if coverage_errors:
        print("\nCoverage Validation Failed:")
        for error in coverage_errors:
            print(" ", error)

        raise ValueError(
            "Candidate dataset does not meet coverage requirements"
        )

    print("Structurally Valid:", len(valid))
    print("Rejected:", len(rejected))

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Write a fresh candidate snapshot.
    # Never overwrite our approved golden dataset.
    with OUTPUT_PATH.open("w", encoding="utf-8") as file:

        for candidate in valid:

            record = {
                "test_id": candidate.test_id,
                "dataset_version": "candidate-v1.0",
                "user_input": candidate.user_input,
                "expected_tool_calls": [
                    {
                        "name": call.name,
                        "args": call.args
                    }
                    for call in candidate.expected_tool_calls
                ]
            }

            file.write(
                json.dumps(record) + "\n"
            )

    print("\nRejected Candidates")

    for item in rejected:
        print(item)

    print("\nValidated Candidates")

    for candidate in valid:

        print(
            f"\n{candidate.test_id}: "
            f"{candidate.user_input}"
        )

        for call in candidate.expected_tool_calls:
            print(
                f"  {call.name}: {call.args}"
            )

    print("\nCandidates saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    asyncio.run(main())

# poetry run python lesson8_3_generate_synthetic_candidates.py
