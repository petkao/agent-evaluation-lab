
from collections import Counter

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition


load_dotenv()


# ============================================================
# 1. Define tools
# ============================================================

@tool
def search_hotels(city: str, max_price: int) -> str:
    """Find hotels in a city below a maximum nightly price."""
    return f"Hotel Sakura in {city}: $180/night."


@tool
def get_weather(city: str) -> str:
    """Get the weather forecast for a city."""
    return f"The weather in {city} is sunny."


# ============================================================
# 2. Configure LLM
# ============================================================

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

llm_with_tools = llm.bind_tools([
    search_hotels,
    get_weather
])


# ============================================================
# 3. Define agent node
# ============================================================

def agent_node(state: MessagesState):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


# ============================================================
# 4. Build and compile LangGraph
# ============================================================

builder = StateGraph(MessagesState)

builder.add_node("agent", agent_node)

builder.add_node(
    "tools",
    ToolNode([search_hotels, get_weather])
)

builder.add_edge(START, "agent")

builder.add_conditional_edges(
    "agent",
    tools_condition,
    {
        "tools": "tools",
        "__end__": END
    }
)

builder.add_edge("tools", "agent")

# Keep graph at module level so other programs can import it.
graph = builder.compile()


# ============================================================
# 5. Main function
# ============================================================

def main():

    # --------------------------------------------------------
    # 5.1 Execute LangGraph agent
    # --------------------------------------------------------

    result = graph.invoke({
        "messages": [
            HumanMessage(
                content=(
                    "Find a hotel in Tokyo under $200 per night "
                    "and check the weather in Tokyo."
                )
            )
        ]
    })

    # --------------------------------------------------------
    # 5.2 Extract actual trajectory
    # --------------------------------------------------------

    actual_trajectory = [
        call["name"]
        for msg in result["messages"]
        if isinstance(msg, AIMessage)
        for call in msg.tool_calls
    ]

    expected_trajectory = [
        "search_hotels",
        "get_weather"
    ]

    # --------------------------------------------------------
    # 5.3 Evaluate trajectory
    # --------------------------------------------------------

    expected_counts = Counter(expected_trajectory)
    actual_counts = Counter(actual_trajectory)

    required_tools_present = all(
        actual_counts[tool_name] >= count
        for tool_name, count in expected_counts.items()
    )

    extra_calls = sum(
        (actual_counts - expected_counts).values()
    )

    missing_calls = sum(
        (expected_counts - actual_counts).values()
    )

    exact_match = (
        expected_trajectory == actual_trajectory
    )

    order_independent_match = (
        expected_counts == actual_counts
    )

    # --------------------------------------------------------
    # 5.4 Print trajectory evaluation
    # --------------------------------------------------------

    print("\nLangGraph Agent Trajectory Evaluation")
    print("-------------------------------------")

    print("Expected:", expected_trajectory)
    print("Actual:", actual_trajectory)
    print("Exact Match:", exact_match)
    print("Order-Independent Match:", order_independent_match)
    print("Required Tools Present:", required_tools_present)
    print("Missing Calls:", missing_calls)
    print("Extra Calls:", extra_calls)

    # --------------------------------------------------------
    # 5.5 Print detailed tool-call history
    # --------------------------------------------------------

    print("\nDetailed Tool Call History")
    print("--------------------------")

    round_number = 0

    for msg in result["messages"]:

        if isinstance(msg, AIMessage) and msg.tool_calls:

            round_number += 1

            print(
                f"\nLLM Tool-Calling Round {round_number}"
            )

            for call in msg.tool_calls:
                print("Tool:", call["name"])
                print("Arguments:", call["args"])
                print("Tool Call ID:", call["id"])

    print(
        "\nTotal LLM Tool-Calling Rounds:",
        round_number
    )


# ============================================================
# 6. Python entry point
# ============================================================

if __name__ == "__main__":
    main()

# poetry run python lesson5_2_langgraph_trajectory.py