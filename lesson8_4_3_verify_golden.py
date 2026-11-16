
import hashlib
import json
from pathlib import Path

dataset_path = Path("datasets/hotel_golden_v2.jsonl")
audit_path = Path("datasets/hotel_golden_v2_approval.json")

records = [
    json.loads(line)
    for line in dataset_path.read_text(encoding="utf-8").splitlines()
    if line.strip()
]

audit = json.loads(audit_path.read_text(encoding="utf-8"))

actual_hash = hashlib.sha256(dataset_path.read_bytes()).hexdigest()

assert len(records) == 10
assert audit["approved_count"] == len(records)
assert audit["status"] == "approved"
assert actual_hash == audit["golden_sha256"]
assert all(record["dataset_version"] == "v2.0" for record in records)
assert len({record["test_id"] for record in records}) == len(records)
assert {record["test_id"] for record in records} == set(
    audit["approved_test_ids"]
)

print("Golden Dataset Verification")
print("---------------------------")
print("Dataset version: v2.0")
print("Records:", len(records))
print("Approval status:", audit["status"])
print("SHA-256 verified:", actual_hash == audit["golden_sha256"])
print("Verification PASSED")

# poetry run python lesson8_4_3_verify_golden.py