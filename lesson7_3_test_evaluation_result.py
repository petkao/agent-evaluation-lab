
from evaluation.models import create_evaluation_result


scores = {
    "tool_call_accuracy": 1.0,
    "tool_call_f1": 1.0
}

result = create_evaluation_result(
    test_id="hotel_001",
    agent_version="v1.0",
    scores=scores
)

print("\nStructured Evaluation Result")
print("----------------------------")
print(result.to_dict())         

# poetry run python lesson7_3_test_evaluation_result.py
