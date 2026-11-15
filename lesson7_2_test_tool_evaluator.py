
import asyncio

from langchain_core.messages import HumanMessage
from ragas.messages import ToolCall

from lesson5_2_langgraph_trajectory import graph
from evaluation.tool_evaluator import evaluate_tool_calls


async def main():

    # 1. Execute the LangGraph agent
    result = await graph.ainvoke({
        "messages": [
            HumanMessage(
                content=(
                    "Find a hotel in Tokyo under $200 per night "
                    "and check the weather in Tokyo."
                )
            )
        ]
    })

    # 2. Define expected tool calls
    expected_calls = [
        ToolCall(
            name="search_hotels",
            args={
                "city": "Tokyo",
                "max_price": 200
            }
        ),
        ToolCall(
            name="get_weather",
            args={
                "city": "Tokyo"
            }
        )
    ]

    # 3. Evaluate actual agent execution
    scores = await evaluate_tool_calls(
        langgraph_messages=result["messages"],
        expected_calls=expected_calls
    )

    # 4. Display evaluation results
    print("\nReusable Tool Evaluator Test")
    print("----------------------------")
    print("Tool Call Accuracy:", scores["tool_call_accuracy"])
    print("Tool Call F1:", scores["tool_call_f1"])


if __name__ == "__main__":
    asyncio.run(main())

# poetry run python lesson7_2_test_tool_evaluator.py
