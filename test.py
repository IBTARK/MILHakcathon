from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition

from langchain_openai import AzureChatOpenAI

from tutor_agent.tools import RAGRetrieveChunks

load_dotenv()

rag_tool = RAGRetrieveChunks()

def assistant(state: MessagesState):
    return {
        "messages": [chat_with_tools.invoke([system_msg] + state["messages"])],
    }

endpoint = "https://hackaton-scv-openai.openai.azure.com/"
model_name = "gpt-4.1-mini"
deployment_name = "gpt-4.1-mini-team6"

api_key = "DpxSOMdIYmTXQRFQ1f7VOoLd2fp4nSE3QAhgEw5fvO8KH9WUbKNEJQQJ99BGACfhMk5XJ3w3AAABACOGk5CC"
api_version = "2024-12-01-preview"

llm = AzureChatOpenAI(
    azure_endpoint   = endpoint,
    azure_deployment = deployment_name,
    api_version      = api_version,
    api_key          = api_key,
    model_name       = "gpt-4.1-mini",
    temperature      = 0.3,
)

chat_with_tools = llm.bind_tools([rag_tool], parallel_tool_calls=False)

query = "¿Qué prefijo de red debe anunciar el encaminador vm_3router?"
msg = HumanMessage(content = query)

system_msg = SystemMessage(
    content=(
        "You are an assistant with access to a retrieval tool called "
        "'rag_retrieve_chunks'. "
        "If the user asks something that might be answered from the "
        "uploaded documents, CALL the tool with the user's question."
    )
)

builder = StateGraph(MessagesState)

# Define nodes: these do the work
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode([rag_tool]))

# Define edges: these determine how the control flow moves
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    # If the latest message requires a tool, route to tools
    # Otherwise, provide a direct response
    tools_condition,
)
builder.add_edge("tools", "assistant")
alfred = builder.compile()

messages = [HumanMessage(content=query)]
response = alfred.invoke({"messages": messages})

for m in response["messages"]:
    m.pretty_print()
