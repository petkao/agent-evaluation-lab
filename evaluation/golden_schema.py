
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ToolCallExpectation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    args: dict[str, Any] = Field(default_factory=dict)


class StateAssertion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field: str = Field(min_length=1)
    operator: Literal[
        "equals",
        "not_equals",
        "exists",
        "not_exists",
        "in",
    ]
    value: Any = None

    @model_validator(mode="after")
    def validate_value(self):
        if self.operator == "in" and not isinstance(
            self.value, (list, tuple, set)
        ):
            raise ValueError("'in' requires a collection value")
        return self


class NodeOrderConstraint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    before: str = Field(min_length=1)
    after: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_order(self):
        if self.before == self.after:
            raise ValueError(
                "A node cannot be ordered before itself"
            )
        return self


class ExpectedBehavior(BaseModel):
    model_config = ConfigDict(extra="forbid")

    required_nodes: list[str] = Field(default_factory=list)
    forbidden_nodes: list[str] = Field(default_factory=list)

    # Node execution ordering constraints
    node_order: list[NodeOrderConstraint] = Field(
        default_factory=list
    )
    
    required_tool_calls: list[ToolCallExpectation] = Field(
        default_factory=list
    )

    state_assertions: list[StateAssertion] = Field(
        default_factory=list
    )

    required_behaviors: list[str] = Field(default_factory=list)
    forbidden_behaviors: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_node_constraints(self):
        overlap = set(self.required_nodes) & set(self.forbidden_nodes)
        if overlap:
            raise ValueError(
                f"Nodes cannot be both required and forbidden: {overlap}"
            )
        return self


class GoldenTestCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    dataset_version: str | None = None
    test_id: str = Field(min_length=1)
    category: str = Field(min_length=1)
    agent_id: str = Field(min_length=1)

    user_input: str = Field(min_length=1)

    initial_state: dict[str, Any] = Field(default_factory=dict)
    mock_tools: dict[str, Any] = Field(default_factory=dict)

    expected: ExpectedBehavior = Field(
        default_factory=ExpectedBehavior
    )
