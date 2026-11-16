
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict
from langchain_openai import ChatOpenAI

load_dotenv()


# --------------------------------------------------
# 1. Structured output models
# --------------------------------------------------

class CandidateToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    city: str
    max_price: int | None

    @property
    def args(self):
        if self.name == "search_hotels":
            return {
                "city": self.city,
                "max_price": self.max_price
            }

        return {"city": self.city}


class CandidateTestCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    test_id: str
    user_input: str
    expected_tool_calls: list[CandidateToolCall]


class CandidateBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    test_cases: list[CandidateTestCase]


# --------------------------------------------------
# 2. Generator instructions
# --------------------------------------------------


GENERATOR_INSTRUCTIONS = """
You generate synthetic evaluation test cases for a travel agent.

The agent supports exactly two tools.

TOOL 1: search_hotels
Signature:
    search_hotels(city: str, max_price: int)

Capabilities:
- Search for hotels in a specified city.
- Filter hotels by a maximum nightly price.
- The maximum price is inclusive.

Limitations:
- No check-in or check-out dates.
- No room availability by date.
- No hotel ratings or amenities filtering.
- No hotel booking.

Valid example:
"Find hotels in Tokyo for $200 or less per night."

TOOL 2: get_weather
Signature:
    get_weather(city: str)

Capabilities:
- Retrieve simulated current weather for a city.

Limitations:
- No future weather forecasts.
- No historical weather.
- No date or time filtering.

Valid example:
"What is the current weather in Paris?"

GENERATION RULES:
- Generate exactly 10 test cases.
- Use unique IDs synthetic_001 through synthetic_010.
- Every request must be fully supported by the tools.
- Do not generate requests requiring unsupported capabilities.
- Use realistic travel-related language.
Generate exactly:
- 4 hotel-only requests
- 3 weather-only requests
- 3 combined hotel-and-weather requests

All 10 requests must use supported capabilities.
Do not generate weekend forecasts or future weather requests.
- Use different cities and budgets.
- All hotel budgets must be positive integers.
- Use inclusive hotel price language:
  "at most", "up to", or "or less".
- Do not use strict price language such as "under" or
  "less than" because the hotel filter is inclusive.
- Do not invent additional tools or arguments.

OUTPUT FORMAT:
Each expected tool call must contain:
- name
- city
- max_price

For search_hotels:
- name = "search_hotels"
- city = requested city
- max_price = requested positive integer budget

For get_weather:
- name = "get_weather"
- city = requested city
- max_price = null

Return the test cases using the required structured schema.
"""


# --------------------------------------------------
# 3. Synthetic candidate generation
# --------------------------------------------------


async def generate_synthetic_candidates():

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7
    )

    structured_llm = llm.with_structured_output(
        CandidateBatch
    )

    generation_plan = [
        ("hotel_only", 4),
        ("weather_only", 3),
        ("combined", 3)
    ]

    all_candidates = []
    next_id = 1

    for category, count in generation_plan:

        category_instructions = f"""
Generate exactly {count} synthetic test cases.

Category: {category}

Category rules:
- hotel_only: call search_hotels only.
- weather_only: call get_weather only.
- combined: call both search_hotels and get_weather.

Use unique cities and realistic requests.
Use supported capabilities only.

Assign test IDs sequentially starting from
synthetic_{next_id:03d}.

Return exactly {count} test cases.
"""

        batch = await structured_llm.ainvoke(
            GENERATOR_INSTRUCTIONS
            + "\n"
            + category_instructions
        )

        if len(batch.test_cases) != count:
            raise ValueError(
                f"{category}: expected {count} cases, "
                f"received {len(batch.test_cases)}"
            )

        # Assign IDs ourselves to guarantee uniqueness.
        for candidate in batch.test_cases:
            candidate.test_id = f"synthetic_{next_id:03d}"
            next_id += 1

        all_candidates.extend(batch.test_cases)

    return all_candidates

