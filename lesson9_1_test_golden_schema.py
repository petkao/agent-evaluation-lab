
from pydantic import ValidationError

from evaluation.golden_schema import GoldenTestCase


def main():
    print("Lesson 9.1 — Generalized Golden Dataset Schema")
    print("-" * 50)

    # --------------------------------------------------
    # Test 1: Valid golden test case
    # --------------------------------------------------

    test_case = {
        "schema_version": "1.0",
        "test_id": "replacement_001",
        "category": "replacement",
        "agent_id": "support_agent",
        "user_input": "My laptop is broken. Order ORD-1001.",
        "initial_state": {
            "status": "new"
        },
        "mock_tools": {
            "get_order": {
                "ORD-1001": {
                    "eligible_for_replacement": True
                }
            }
        },
        "expected": {
            "required_nodes": [
                "intent_router",
                "order_lookup",
                "replacement"
            ],
            "forbidden_nodes": [
                "refund"
            ],
            "node_order": [
                {
                    "before": "intent_router",
                    "after": "order_lookup"
                },
                {
                    "before": "order_lookup",
                    "after": "replacement"
                }
            ],
            "required_tool_calls": [
                {
                    "name": "get_order",
                    "args": {
                        "order_id": "ORD-1001"
                    }
                }
            ],
            "state_assertions": [
                {
                    "field": "intent",
                    "operator": "equals",
                    "value": "replacement"
                }
            ],
            "required_behaviors": [
                "Verify replacement eligibility",
                "Request customer confirmation"
            ],
            "forbidden_behaviors": [
                "Issue replacement without confirmation"
            ]
        }
    }

    golden = GoldenTestCase.model_validate(test_case)

    print("Test ID:", golden.test_id)
    print("Agent:", golden.agent_id)
    print("Category:", golden.category)

    print("Required nodes:", golden.expected.required_nodes)
    print("Forbidden nodes:", golden.expected.forbidden_nodes)

    print("Required tools:", [
        tool.name
        for tool in golden.expected.required_tool_calls
    ])

    print("Node order constraints:", [
        f"{item.before} -> {item.after}"
        for item in golden.expected.node_order
    ])

    print("\nSchema validation: PASS")

    # --------------------------------------------------
    # Test 2: Conflicting required/forbidden nodes
    # --------------------------------------------------

    invalid_case = golden.model_dump()

    invalid_case["expected"]["required_nodes"].append("refund")

    try:
        GoldenTestCase.model_validate(invalid_case)
    except ValidationError:
        print("Conflicting node constraints: correctly rejected")
    else:
        raise AssertionError(
            "Expected conflicting node constraints to be rejected"
        )

    # --------------------------------------------------
    # Test 3: Invalid node ordering
    # --------------------------------------------------

    invalid_order = golden.model_dump()

    invalid_order["expected"]["node_order"] = [
        {
            "before": "intent_router",
            "after": "intent_router"
        }
    ]

    try:
        GoldenTestCase.model_validate(invalid_order)
    except ValidationError:
        print("Invalid node ordering: correctly rejected")
    else:
        raise AssertionError(
            "Expected invalid node ordering to be rejected"
        )

    print("\nAll Lesson 9.1 tests PASSED")


if __name__ == "__main__":
    main()

#poetry run python lesson9_1_test_golden_schema.py
