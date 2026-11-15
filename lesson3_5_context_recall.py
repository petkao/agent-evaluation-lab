
import asyncio

from dotenv import load_dotenv
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import ContextRecall


async def main():
    load_dotenv()

    client = AsyncOpenAI()

    evaluator_llm = llm_factory(
        "gpt-4o-mini",
        client=client
    )

    scorer = ContextRecall(llm=evaluator_llm)

    question = (
        "What is Hotel Sakura's nightly price, "
        "and does it include breakfast?"
    )

    reference = (
        "Hotel Sakura costs $180 per night, "
        "and breakfast is included."
    )

    '''
    retrieved_contexts = [
        "Hotel Sakura costs $180 per night.",
        "Hotel Sakura has 120 rooms."
    ]
    '''

    retrieved_contexts = [
    "Hotel Sakura costs $180 per night.",
    "Hotel Sakura has 120 rooms.",
    "Hotel Sakura includes complimentary breakfast."
]

    result = await scorer.ascore(
        user_input=question,
        reference=reference,
        retrieved_contexts=retrieved_contexts
    )

    print("\nRAGAS Context Recall Evaluation")
    print("-------------------------------")
    print("Question:", question)
    print("Reference:", reference)
    print("Retrieved Contexts:", retrieved_contexts)
    print("Context Recall Score:", result.value)


if __name__ == "__main__":
    asyncio.run(main())

# poetry run python lesson3_5_context_recall.py