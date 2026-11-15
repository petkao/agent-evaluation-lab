
from ragas.metrics.collections import (
    ToolCallAccuracy,
    ToolCallF1,
)

from evaluation.adapters import convert_messages


async def evaluate_tool_calls(
    langgraph_messages,
    expected_calls
):
    """
    Evaluate LangGraph tool calls using RAGAS.

    Parameters:
        langgraph_messages:
            Messages returned by a LangGraph execution.

        expected_calls:
            Expected RAGAS ToolCall objects.

    Returns:
        Dictionary containing evaluation scores.
    """

    # 1. Convert LangGraph messages into RAGAS format
    conversation = convert_messages(langgraph_messages)

    # 2. Initialize deterministic RAGAS metrics
    accuracy_metric = ToolCallAccuracy(strict_order=True)
    f1_metric = ToolCallF1()

    # 3. Calculate tool-call accuracy
    accuracy = await accuracy_metric.ascore(
        user_input=conversation,
        reference_tool_calls=expected_calls
    )

    # 4. Calculate tool-call F1
    f1 = await f1_metric.ascore(
        user_input=conversation,
        reference_tool_calls=expected_calls
    )

    # 5. Return structured scores
    return {
        "tool_call_accuracy": accuracy.value,
        "tool_call_f1": f1.value
    }
