
import re

from evaluation.synthetic_generator import CandidateTestCase


ALLOWED_TOOLS = {
    "search_hotels": {"city", "max_price"},
    "get_weather": {"city"}
}


def validate_candidate(candidate: CandidateTestCase):

    errors = []

    if not candidate.user_input.strip():
        errors.append("Empty user input")

    if not re.fullmatch(
        r"synthetic_\d{3}",
        candidate.test_id
    ):
        errors.append("Invalid test ID format")

    if not candidate.expected_tool_calls:
        errors.append("No expected tool calls")

    for call in candidate.expected_tool_calls:

        if call.name not in ALLOWED_TOOLS:
            errors.append(
                f"Unsupported tool: {call.name}"
            )
            continue

        required_args = ALLOWED_TOOLS[call.name]

        if set(call.args.keys()) != required_args:
            errors.append(
                f"Invalid arguments for {call.name}: {call.args}"
            )
            continue

        city = call.args["city"]

        if not isinstance(city, str) or not city.strip():
            errors.append("Invalid city")

        if call.name == "search_hotels":

            budget = call.args["max_price"]

            if (
                type(budget) is not int
                or budget <= 0
            ):
                errors.append("Invalid hotel budget")

    return errors


def validate_batch(candidates):

    valid = []
    rejected = []
    seen_ids = set()
    seen_requests = set()

    for candidate in candidates:

        errors = validate_candidate(candidate)

        errors.extend(
            validate_capabilities(candidate)
        )

        normalized_request = (
            " ".join(candidate.user_input.lower().split())
        )

        if candidate.test_id in seen_ids:
            errors.append("Duplicate test ID")

        if normalized_request in seen_requests:
            errors.append("Duplicate user request")

        seen_ids.add(candidate.test_id)
        seen_requests.add(normalized_request)

        if errors:
            rejected.append({
                "test_id": candidate.test_id,
                "errors": errors
            })
        else:
            valid.append(candidate)

    return valid, rejected

def validate_capabilities(candidate):
    errors = []

    request = candidate.user_input.lower()

    unsupported_weather_terms = [
        "tomorrow",
        "next week",
        "this weekend",
        "next weekend",
        "yesterday",
        "last week"
    ]

    uses_weather = any(
        call.name == "get_weather"
        for call in candidate.expected_tool_calls
    )

    if uses_weather:
        for term in unsupported_weather_terms:
            if term in request:
                errors.append(
                    f"Unsupported weather time constraint: {term}"
                )

    return errors


def validate_dataset_coverage(candidates):

    counts = {
        "hotel_only": 0,
        "weather_only": 0,
        "combined": 0
    }

    for candidate in candidates:

        tool_names = {
            call.name
            for call in candidate.expected_tool_calls
        }

        if tool_names == {"search_hotels"}:
            counts["hotel_only"] += 1

        elif tool_names == {"get_weather"}:
            counts["weather_only"] += 1

        elif tool_names == {
            "search_hotels",
            "get_weather"
        }:
            counts["combined"] += 1

    expected = {
        "hotel_only": 4,
        "weather_only": 3,
        "combined": 3
    }

    errors = []

    for category, expected_count in expected.items():

        if counts[category] != expected_count:
            errors.append(
                f"{category}: expected {expected_count}, "
                f"found {counts[category]}"
            )

    return counts, errors



