
from typing import TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END


load_dotenv()


# -------------------------------------------------
# 1. Define tools available to the agent
# -------------------------------------------------

@tool
def search_hotels(city: str, max_price: int) -> str:
    """Search hotels in a city with a maximum nightly price."""
    return f"Searching hotels in {city} under ${max_price}"


@tool
def get_weather(city: str) -> str:
    """Get the weather forecast for a city."""
    return f"Weather forecast for {city}"


# -------------------------------------------------
# 2. Define LangGraph state
# -------------------------------------------------

class AgentState(TypedDict):
    question: str
    tool_calls: list


# -------------------------------------------------
# 3. Define the agent node
# -------------------------------------------------

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

llm_with_tools = llm.bind_tools(
    [search_hotels, get_weather]
)


def agent_node(state: AgentState):
    response = llm_with_tools.invoke([
        HumanMessage(content=state["question"])
    ])

    return {
        "tool_calls": response.tool_calls
    }


# -------------------------------------------------
# 4. Build LangGraph
# -------------------------------------------------

builder = StateGraph(AgentState)

builder.add_node("agent", agent_node)

builder.add_edge(START, "agent")
builder.add_edge("agent", END)

graph = builder.compile()


# -------------------------------------------------
# 5. Evaluate the generated tool call
# -------------------------------------------------

# question = "Find hotels in Tokyo under $200 per night."
# question = "Find hotels in Tokyo under $200 per night."
question = "What is the weather forecast for Tokyo?"

result = graph.invoke({
    "question": question,
    "tool_calls": []
})

print("\nGenerated Tool Calls")
print("--------------------")
print(result["tool_calls"])

# expected_tool = "search_hotels"
expected_tool = "get_weather"

# expected_args = {
#     "city": "Tokyo",
#     "max_price": 200
# }

expected_args = {
    "city": "Tokyo"
}

if result["tool_calls"]:
    actual_call = result["tool_calls"][0]

    tool_correct = (
        actual_call["name"] == expected_tool
    )

    actual_args = actual_call["args"]

    argument_results = {
        key: actual_args.get(key) == value
        for key, value in expected_args.items()
    }

    argument_accuracy = (
        sum(argument_results.values())
        / len(argument_results)
    )

    all_correct = (
        tool_correct
        and all(argument_results.values())
    )

    print("\nTool Call Evaluation")
    print("--------------------")
    print("Tool Correct:", tool_correct)
    print("Argument Results:", argument_results)
    print("Argument Accuracy:", argument_accuracy)
    print("All Correct:", all_correct)

else:
    print("Agent did not generate a tool call.")

# poetry run python lesson4_2_langgraph_tool_evaluation.py
