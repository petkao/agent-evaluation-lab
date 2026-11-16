
import json

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field
from langchain_openai import ChatOpenAI

load_dotenv()


class SemanticReview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    test_id: str
    is_valid: bool
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str


REVIEW_INSTRUCTIONS = """
You are reviewing synthetic test cases for an AI travel agent.

The agent has exactly two tools:

1. search_hotels(city: str, max_price: int)
   - Searches hotels in a city.
   - Maximum nightly price is inclusive.
   - Does not support booking, dates, or availability.

2. get_weather(city: str)
   - Returns simulated current weather.
   - Does not support future forecasts or historical weather.

Review whether the expected tool calls correctly represent
the user's request.

Check:
- Correct tool selection.
- Correct city.
- Correct hotel budget.
- Correct number of tool calls.
- No missing or unnecessary tool calls.
- No unsupported capabilities.
- No contradictions between user request and tool arguments.

If the user requests an unsupported capability,
mark the test case invalid.

Return:
- test_id
- is_valid
- confidence (0.0 to 1.0)
- reason

Do not assume the expected tool calls are correct.
"""


async def review_candidate(candidate: dict) -> SemanticReview:

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

    reviewer = llm.with_structured_output(SemanticReview)

    candidate_text = json.dumps(
        candidate,
        indent=2
    )

    result = await reviewer.ainvoke(
        [
            ("system", REVIEW_INSTRUCTIONS),
            (
                "human",
                "Review this candidate:\n" + candidate_text
            )
        ]
    )

    return result
