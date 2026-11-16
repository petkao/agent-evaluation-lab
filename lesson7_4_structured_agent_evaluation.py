
import asyncio
import json

from langchain_core.messages import HumanMessage
from ragas.messages import ToolCall

from lesson5_2_langgraph_trajectory import graph
from evaluation.tool_evaluator import evaluate_tool_calls
from evaluation.models import create_evaluation_result


async def main():

    # --------------------------------------------------------
    # 1. Execute the real LangGraph agent
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # 2. Define expected tool calls
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # 3. Evaluate actual agent execution
    # --------------------------------------------------------

    scores = await evaluate_tool_calls(
        langgraph_messages=result["messages"],
        expected_calls=expected_calls
    )

    # --------------------------------------------------------
    # 4. Create structured evaluation record
    # --------------------------------------------------------

    evaluation_result = create_evaluation_result(
        test_id="hotel_001",
        agent_version="v1.0",
        scores=scores
    )

    # --------------------------------------------------------
    # 5. Display structured JSON
    # --------------------------------------------------------

    print("\nStructured LangGraph Evaluation")
    print("-------------------------------")

    print(
        json.dumps(
            evaluation_result.to_dict(),
            indent=2
        )
    )


if __name__ == "__main__":
    asyncio.run(main())

# poetry run python lesson7_4_structured_agent_evaluation.py