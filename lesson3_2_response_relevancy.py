import asyncio

from dotenv import load_dotenv
from openai import AsyncOpenAI

from ragas.llms import llm_factory
from ragas.embeddings import embedding_factory
from ragas.metrics.collections import AnswerRelevancy


async def main():
    load_dotenv()

    client = AsyncOpenAI()

    evaluator_llm = llm_factory(
        "gpt-4o-mini",
        client=client
    )

    evaluator_embeddings = embedding_factory(
        "openai",
        model="text-embedding-3-small",
        client=client
    )

    scorer = AnswerRelevancy(
        llm=evaluator_llm,
        embeddings=evaluator_embeddings
    )

    question = "What is the nightly price of Hotel Sakura?"

    '''
    response = (
        "Hotel Sakura is located in Tokyo "
        "and offers comfortable rooms."
    )
    '''
    response = "Hotel Sakura costs $180 per night."

    result = await scorer.ascore(
        user_input=question,
        response=response
    )

    print("\nResponse Relevancy Evaluation")
    print("-----------------------------")
    print("Question:", question)
    print("Agent Response:", response)
    print("Response Relevancy Score:", result.value)


if __name__ == "__main__":
    asyncio.run(main())

# poetry run python lesson3_2_response_relevancy.py