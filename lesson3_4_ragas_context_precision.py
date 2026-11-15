
import asyncio

from dotenv import load_dotenv
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import ContextPrecision


async def main():
    load_dotenv()

    client = AsyncOpenAI()

    evaluator_llm = llm_factory(
        "gpt-4o-mini",
        client=client
    )

    scorer = ContextPrecision(llm=evaluator_llm)

    question = "What is the nightly price of Hotel Sakura?"

    reference = "Hotel Sakura costs $180 per night."

    '''
    retrieved_contexts = [
        "Hotel Sakura offers free parking.",
        "Hotel Sakura has 120 rooms.",
        "Hotel Sakura costs $180 per night."
    ]
    '''

    retrieved_contexts = [
    "Hotel Sakura costs $180 per night.",
    "Hotel Sakura offers free parking.",
    "Hotel Sakura has 120 rooms."
]

    result = await scorer.ascore(
        user_input=question,
        reference=reference,
        retrieved_contexts=retrieved_contexts
    )

    print("\nRAGAS Context Precision Evaluation")
    print("----------------------------------")
    print("Question:", question)
    print("Reference:", reference)

    for rank, context in enumerate(retrieved_contexts, start=1):
        print(f"Rank {rank}: {context}")

    print("Context Precision Score:", result.value)


if __name__ == "__main__":
    asyncio.run(main())

# poetry run python lesson3_4_ragas_context_precision.py
