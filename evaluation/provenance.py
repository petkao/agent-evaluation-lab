
import hashlib
import json


def candidate_sha256(candidate: dict) -> str:
    """
    Compute a deterministic fingerprint of a candidate.
    """
    canonical = json.dumps(
        candidate,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False
    )

    return hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest()
