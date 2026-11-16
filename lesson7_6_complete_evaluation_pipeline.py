
import asyncio
import json

from langchain_core.messages import HumanMessage
from ragas.messages import ToolCall

from lesson5_2_langgraph_trajectory import graph

from evaluation.tool_evaluator import evaluate_tool_calls
from evaluation.models import create_evaluation_result
from evaluation.storage import (
    save_evaluation_result,
    load_evaluation_results,
)


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
    # 3. Evaluate actual tool calls with RAGAS
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
    # 5. Persist evaluation result to JSONL
    # --------------------------------------------------------

    save_evaluation_result(evaluation_result)

    # --------------------------------------------------------
    # 6. Load saved evaluation history
    # --------------------------------------------------------

    history = load_evaluation_results()

    # --------------------------------------------------------
    # 7. Display current result and history count
    # --------------------------------------------------------

    print("\nComplete LangGraph Evaluation Pipeline")
    print("--------------------------------------")

    print(
        json.dumps(
            evaluation_result.to_dict(),
            indent=2
        )
    )

    print("\nTotal Stored Evaluation Records:", len(history))


if __name__ == "__main__":
    asyncio.run(main())

# poetry run python lesson7_6_complete_evaluation_pipeline.py
