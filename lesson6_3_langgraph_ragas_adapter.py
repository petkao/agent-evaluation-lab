
import asyncio

from langchain_core.messages import (
    HumanMessage as LangGraphHumanMessage,
    AIMessage as LangGraphAIMessage,
)

from ragas.messages import (
    HumanMessage as RagasHumanMessage,
    AIMessage as RagasAIMessage,
    ToolCall as RagasToolCall,
)

from ragas.metrics.collections import (
    ToolCallAccuracy,
    ToolCallF1,
)


def convert_messages(langgraph_messages):
    """Convert LangGraph messages into RAGAS messages.

    This first adapter handles user messages and AI tool calls.
    """

    ragas_messages = []

    for msg in langgraph_messages:

        if isinstance(msg, LangGraphHumanMessage):
            ragas_messages.append(
                RagasHumanMessage(content=msg.content)
            )

        elif isinstance(msg, LangGraphAIMessage):
            tool_calls = [
                RagasToolCall(
                    name=call["name"],
                    args=call["args"]
                )
                for call in msg.tool_calls
            ]

            ragas_messages.append(
                RagasAIMessage(
                    content=msg.content or "",
                    tool_calls=tool_calls
                )
            )

    return ragas_messages


async def evaluate_tool_calls(langgraph_messages, expected_calls):

    conversation = convert_messages(langgraph_messages)

    accuracy_metric = ToolCallAccuracy(strict_order=True)
    f1_metric = ToolCallF1()

    accuracy = await accuracy_metric.ascore(
        user_input=conversation,
        reference_tool_calls=expected_calls
    )

    f1 = await f1_metric.ascore(
        user_input=conversation,
        reference_tool_calls=expected_calls
    )

    return {
        "tool_call_accuracy": accuracy.value,
        "tool_call_f1": f1.value
    }


async def main():

    # Simulated LangGraph output for testing the adapter.
    # In the next step, we'll use actual graph.invoke() output.

    langgraph_messages = [
        LangGraphHumanMessage(
            content="Find a hotel in Tokyo under $200 and check the weather."
        ),
        LangGraphAIMessage(
            content="",
            tool_calls=[
                {
                    "name": "search_hotels",
                    "args": {"city": "Tokyo", "max_price": 200},
                    "id": "call_001"
                },
                {
                    "name": "get_weather",
                    "args": {"city": "Tokyo"},
                    "id": "call_002"
                }
            ]
        )
    ]

    expected_calls = [
        RagasToolCall(
            name="search_hotels",
            args={"city": "Tokyo", "max_price": 200}
        ),
        RagasToolCall(
            name="get_weather",
            args={"city": "Tokyo"}
        )
    ]

    scores = await evaluate_tool_calls(
        langgraph_messages,
        expected_calls
    )

    print("\nLangGraph to RAGAS Evaluation")
    print("-----------------------------")
    print("Tool Call Accuracy:", scores["tool_call_accuracy"])
    print("Tool Call F1:", scores["tool_call_f1"])


if __name__ == "__main__":
    asyncio.run(main())

# poetry run python lesson6_3_langgraph_ragas_adapter.py