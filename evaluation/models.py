
from dataclasses import dataclass, asdict
from datetime import datetime, timezone


@dataclass
class EvaluationResult:
    test_id: str
    agent_version: str
    tool_call_accuracy: float
    tool_call_f1: float
    timestamp: str

    def to_dict(self):
        return asdict(self)


def create_evaluation_result(
    test_id: str,
    agent_version: str,
    scores: dict
) -> EvaluationResult:

    return EvaluationResult(
        test_id=test_id,
        agent_version=agent_version,
        tool_call_accuracy=scores["tool_call_accuracy"],
        tool_call_f1=scores["tool_call_f1"],
        timestamp=datetime.now(timezone.utc).isoformat()
    )


