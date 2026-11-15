
import asyncio
from dotenv import load_dotenv
from openai import AsyncOpenAI

from ragas.llms import llm_factory
from ragas.metrics.collections import AgentGoalAccuracyWithReference

from ragas.messages import HumanMessage, AIMessage

load_dotenv()


async def main():
    client = AsyncOpenAI()

    evaluator_llm = llm_factory(
        "gpt-4o-mini",
        client=client
    )

    metric = AgentGoalAccuracyWithReference(
        llm=evaluator_llm
    )

    user_input = (
        "Find a hotel in Tokyo under $200 per night "
        "and check the weather in Tokyo."
    )

    reference = (
        "The agent should identify a hotel in Tokyo "
        "costing less than $200 per night and provide "
        "the weather in Tokyo."
    )

    responses = {
        "Scenario A": (
            "Hotel Sakura in Tokyo costs $180 per night. "
            "The weather in Tokyo is sunny."
        ),
        "Scenario B": (
            "Hotel Sakura in Tokyo costs $300 per night. "
            "The weather in Tokyo is sunny."
        )
    }

    for scenario, response in responses.items():

        conversation = [
            HumanMessage(content=user_input),
            AIMessage(content=response)
        ]

        result = await metric.ascore(
            user_input=conversation,
            reference=reference
        )

        print(f"\n{scenario}")
        print("Agent Response:", response)
        print("Goal Accuracy:", result.value)


if __name__ == "__main__":
    asyncio.run(main())

# poetry run python lesson6_1_agent_goal_accuracy.py
