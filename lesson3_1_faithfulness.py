import asyncio

from dotenv import load_dotenv
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import Faithfulness


async def main():
    load_dotenv()

    client = AsyncOpenAI()

    evaluator_llm = llm_factory(
        "gpt-4o-mini",
        client=client
    )

    scorer = Faithfulness(llm=evaluator_llm)

    question = "What is the nightly price of Hotel Sakura?"

    context = [
        "Hotel Sakura costs $180 per night."
    ]

    # response = "Hotel Sakura costs $350 per night."
    response = "Hotel Sakura costs $180 per night."

    result = await scorer.ascore(
        user_input=question,
        response=response,
        retrieved_contexts=context
    )

    print("\nFaithfulness Evaluation")
    print("-----------------------")
    print("Question:", question)
    print("Retrieved Context:", context)
    print("Agent Response:", response)
    print("Faithfulness Score:", result.value)


if __name__ == "__main__":
    asyncio.run(main()) 

# python lesson3_1_faithfulness.py