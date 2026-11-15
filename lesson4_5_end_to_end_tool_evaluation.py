
import json

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition


load_dotenv()


# 1. Define a tool returning structured data
@tool
def search_hotels(city: str, max_price: int) -> str:
    """Find hotels in a city within a maximum nightly price."""

    hotel = {
        "name": "Hotel Sakura",
        "city": city,
        "price": 180
    }

    return json.dumps(hotel)


# 2. Configure the LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

llm_with_tools = llm.bind_tools([search_hotels])


# 3. Agent node
def agent_node(state: MessagesState):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


# 4. Build LangGraph
builder = StateGraph(MessagesState)

builder.add_node("agent", agent_node)
builder.add_node("tools", ToolNode([search_hotels]))

builder.add_edge(START, "agent")

builder.add_conditional_edges(
    "agent",
    tools_condition,
    {
        "tools": "tools",
        "__end__": END
    }
)

builder.add_edge("tools", END)

graph = builder.compile()


# 5. Execute graph
result = graph.invoke({
    "messages": [
        HumanMessage(
            content="Find hotels in Tokyo under $200 per night."
        )
    ]
})


# 6. Extract messages
ai_messages = [
    msg for msg in result["messages"]
    if isinstance(msg, AIMessage)
]

tool_messages = [
    msg for msg in result["messages"]
    if isinstance(msg, ToolMessage)
]


# 7. Expected behavior
expected_tool = "search_hotels"

expected_args = {
    "city": "Tokyo",
    "max_price": 200
}


# 8. Evaluate tool selection and arguments
tool_calls = [
    call
    for msg in ai_messages
    for call in msg.tool_calls
]

tool_correct = (
    len(tool_calls) == 1
    and tool_calls[0]["name"] == expected_tool
)

argument_results = {}

if tool_calls:
    actual_args = tool_calls[0]["args"]

    argument_results = {
        key: actual_args.get(key) == value
        for key, value in expected_args.items()
    }

argument_accuracy = (
    sum(argument_results.values()) / len(expected_args)
    if argument_results else 0.0
)


# 9. Evaluate tool execution
execution_success = (
    len(tool_messages) == 1
    and tool_messages[0].status == "success"
)


# 10. Evaluate output correctness
output_correct = False

if execution_success:
    try:
        hotel = json.loads(tool_messages[0].content)

        output_correct = (
            hotel["city"] == expected_args["city"]
            and hotel["price"] <= expected_args["max_price"]
        )

    except (json.JSONDecodeError, KeyError, TypeError):
        output_correct = False


# 11. Print evaluation results
print("\nEnd-to-End Tool Evaluation")
print("--------------------------")

print("Tool Selection Correct:", tool_correct)
print("Argument Results:", argument_results)
print("Argument Accuracy:", argument_accuracy)
print("Execution Success:", execution_success)
print("Output Correct:", output_correct)

overall_success = (
    tool_correct
    and argument_accuracy == 1.0
    and execution_success
    and output_correct
)

print("Overall Success:", overall_success)

# poetry run python lesson4_5_end_to_end_tool_evaluation.py
