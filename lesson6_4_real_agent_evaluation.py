
import asyncio

from langchain_core.messages import HumanMessage

from ragas.messages import ToolCall as RagasToolCall

from lesson5_2_langgraph_trajectory import graph

from lesson6_3_langgraph_ragas_adapter import (
    convert_messages,
    evaluate_tool_calls,
)


async def main():

    # 1. Execute the real LangGraph agent
    result = graph.invoke({
        "messages": [
            HumanMessage(
                content=(
                    "Find a hotel in Tokyo under $200 per night "
                    "and check the weather in Tokyo."
                )
            )
        ]
    })

    langgraph_messages = result["messages"]

    # 2. Convert messages to RAGAS format
    ragas_messages = convert_messages(langgraph_messages)

    print("\nConverted RAGAS Messages")
    print("------------------------")

    for msg in ragas_messages:
        print(type(msg).__name__)

    # 3. Define expected tool calls
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

    # 4. Evaluate the actual agent execution
    scores = await evaluate_tool_calls(
        langgraph_messages,
        expected_calls
    )

    print("\nReal LangGraph Agent Evaluation")
    print("-------------------------------")
    print("Tool Call Accuracy:", scores["tool_call_accuracy"])
    print("Tool Call F1:", scores["tool_call_f1"])


if __name__ == "__main__":
    asyncio.run(main())

# poetry run python lesson6_4_real_agent_evaluation.py
