def evaluate_tool_call(expected, actual):
    tool_correct = expected["tool"] == actual["tool"]

    expected_args = expected["arguments"]
    actual_args = actual["arguments"]

    correct_args = sum(
        actual_args.get(key) == value
        for key, value in expected_args.items()
    )

    argument_accuracy = correct_args / len(expected_args)

    return {
        "tool_correct": tool_correct,
        "argument_accuracy": argument_accuracy,
        "all_correct": tool_correct and argument_accuracy == 1.0
    }


expected = {
    "tool": "search_hotels",
    "arguments": {
        "city": "Tokyo",
        "max_price": 200
    }
}

actual = {
    "tool": "search_hotels",
    "arguments": {
        "city": "Tokyo",
        "max_price": 300
    }
}

result = evaluate_tool_call(expected, actual)

print("Evaluation Result:")
print(result)       

# python lesson1_tool_evaluation.py