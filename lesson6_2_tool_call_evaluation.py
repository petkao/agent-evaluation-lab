
import asyncio

from ragas.messages import HumanMessage, AIMessage, ToolCall
from ragas.metrics.collections import ToolCallAccuracy, ToolCallF1


async def main():

    expected_calls = [
        ToolCall(
            name="search_hotels",
            args={"city": "Tokyo", "max_price": 200}
        ),
        ToolCall(
            name="get_weather",
            args={"city": "Tokyo"}
        )
    ]

    scenarios = {
        "Scenario A - Correct": [
            ToolCall(
                name="search_hotels",
                args={"city": "Tokyo", "max_price": 200}
            ),
            ToolCall(
                name="get_weather",
                args={"city": "Tokyo"}
            )
        ],

        "Scenario B - Wrong Price": [
            ToolCall(
                name="search_hotels",
                args={"city": "Tokyo", "max_price": 500}
            ),
            ToolCall(
                name="get_weather",
                args={"city": "Tokyo"}
            )
        ]
    }

    accuracy_metric = ToolCallAccuracy(strict_order=True)
    f1_metric = ToolCallF1()

    for scenario, actual_calls in scenarios.items():

        conversation = [
            HumanMessage(
                content="Find a hotel in Tokyo under $200 and check the weather."
            ),
            AIMessage(
                content="Calling tools.",
                tool_calls=actual_calls
            )
        ]

        accuracy = await accuracy_metric.ascore(
            user_input=conversation,
            reference_tool_calls=expected_calls
        )

        f1 = await f1_metric.ascore(
            user_input=conversation,
            reference_tool_calls=expected_calls
        )

        print(f"\n{scenario}")
        print("Tool Call Accuracy:", accuracy.value)
        print("Tool Call F1:", f1.value)


if __name__ == "__main__":
    asyncio.run(main())

# poetry run python lesson6_2_tool_call_evaluation.py
