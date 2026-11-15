
def evaluate_trajectory(expected, actual):
    # Compare the complete sequence
    exact_match = expected == actual

    # Check whether all required tools were called
    required_tools_present = all(
        tool in actual for tool in expected
    )

    # Count unnecessary tool calls
    remaining = actual.copy()

    for tool in expected:
        if tool in remaining:
            remaining.remove(tool)

    extra_calls = len(remaining)

    # Calculate efficiency
    efficiency = (
        len(expected) / len(actual)
        if actual else 0.0
    )

    return {
        "exact_match": exact_match,
        "required_tools_present": required_tools_present,
        "extra_calls": extra_calls,
        "efficiency": efficiency
    }


expected = [
    "search_hotels",
    "get_weather"
]

'''
actual = [
    "search_hotels",
    "search_hotels",
    "get_weather"
]
'''

actual = [
    "get_weather",
    "search_hotels"
]

result = evaluate_trajectory(expected, actual)

print("\nAgent Trajectory Evaluation")
print("---------------------------")
print("Expected:", expected)
print("Actual:", actual)
print("Evaluation:", result)

# poetry run python lesson5_1_trajectory_evaluation.py