
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool

from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition


load_dotenv()


# ------------------------------------------
# 1. Define a tool
# ------------------------------------------

@tool
def search_hotels(city: str, max_price: int) -> str:
    """Search for hotels in a city below a maximum nightly price."""

    return (
        f"Found Hotel Sakura in {city} "
        f"for $180 per night (limit ${max_price})."
    )


# ------------------------------------------
# 2. Configure the LLM
# ------------------------------------------

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

llm_with_tools = llm.bind_tools([search_hotels])


# ------------------------------------------
# 3. Define the agent node
# ------------------------------------------

def agent_node(state: MessagesState):
    response = llm_with_tools.invoke(
        state["messages"]
    )

    return {
        "messages": [response]
    }


# ------------------------------------------
# 4. Build LangGraph
# ------------------------------------------

builder = StateGraph(MessagesState)

builder.add_node("agent", agent_node)

builder.add_node(
    "tools",
    ToolNode([search_hotels])
)

builder.add_edge(START, "agent")

builder.add_conditional_edges(
    "agent",
    tools_condition
)

builder.add_edge("tools", END)

graph = builder.compile()


# ------------------------------------------
# 5. Execute the agent
# ------------------------------------------

result = graph.invoke({
    "messages": [
        HumanMessage(
            content="Find hotels in Tokyo under $200 per night."
        )
    ]
})


# ------------------------------------------
# 6. Evaluate tool execution
# ------------------------------------------

print("\nLangGraph Tool Execution")
print("------------------------")

tool_messages = [
    msg for msg in result["messages"]
    if isinstance(msg, ToolMessage)
]

for msg in tool_messages:
    print("Tool Name:", msg.name)
    print("Tool Result:", msg.content)
    print("Tool Call ID:", msg.tool_call_id)
    print("Execution Error:", msg.status == "error")

if tool_messages:
    execution_success = all(
        msg.status != "error"
        for msg in tool_messages
    )

    print("\nTool Execution Evaluation")
    print("-------------------------")
    print("Execution Success:", execution_success)
else:
    print("No tool execution detected.")

print("\nComplete LangGraph Message History")
print("----------------------------------")

for index, msg in enumerate(result["messages"], start=1):
    print(f"\nMessage {index}")
    print("Type:", type(msg).__name__)
    print("Content:", msg.content)

    if hasattr(msg, "tool_calls") and msg.tool_calls:
        print("Tool Calls:", msg.tool_calls)

    if isinstance(msg, ToolMessage):
        print("Tool Call ID:", msg.tool_call_id)
        print("Status:", msg.status)

# poetry run python lesson4_3_tool_execution.py