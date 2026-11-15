
from langchain_core.messages import (
    HumanMessage as LangGraphHumanMessage,
    AIMessage as LangGraphAIMessage,
)

from ragas.messages import (
    HumanMessage as RagasHumanMessage,
    AIMessage as RagasAIMessage,
    ToolCall as RagasToolCall,
)


def convert_messages(langgraph_messages):
    """
    Convert LangGraph messages into RAGAS messages.

    Supported:
        - HumanMessage
        - AIMessage
        - AI tool calls

    ToolMessage support will be added later.
    """

    ragas_messages = []

    for msg in langgraph_messages:

        if isinstance(msg, LangGraphHumanMessage):

            ragas_messages.append(
                RagasHumanMessage(content=msg.content)
            )

        elif isinstance(msg, LangGraphAIMessage):

            tool_calls = [
                RagasToolCall(
                    name=call["name"],
                    args=call["args"]
                )
                for call in msg.tool_calls
            ]

            ragas_messages.append(
                RagasAIMessage(
                    content=msg.content or "",
                    tool_calls=tool_calls
                )
            )

    return ragas_messages
