import os
from pydantic import BaseModel
from typing import Any, Literal, Union

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph import MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import SystemMessage, HumanMessage, AnyMessage
from langchain_core.runnables.config import RunnableConfig
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from langgraph.prebuilt import ToolNode

from memory_agent.memory_agent import MemoryAgent
import tutor_agent.tutor_agent_prompts as prompts
from tutor_agent.tools import RAGRetrieveChunks


class TutorAgent:

    SYSTEM_PROMPT_TUTOR_AGENT = prompts.SYSTEM_PROMPT_TUTOR_AGENT


    def __init__(
        self
    ):
        # Define the model to be used
        self.llm = ChatOpenAI(model = "gpt-4o-mini", api_key = os.getenv("OPENAI_API_KEY"))

        # Define the tools
        self.tools = [RAGRetrieveChunks()]

        # Bind the tools to the model
        self.llm_with_tools = self.llm.bind_tools(self.tools)

        # Checkpointers and memory store
        self.checkpointer = MemorySaver()
        self.across_thread_memory = InMemoryStore()

        # Define the memory agent
        self.memory_agent = MemoryAgent(self.checkpointer, self.across_thread_memory)

        self.graph = self.build_graph()

    # Nodes
    def empty_node(
        self,
        state: MessagesState
    ):
        return {}

    async def assistant(
        self,
        state: MessagesState,
        config: RunnableConfig, 
        store: BaseStore
    ):
        system_message = SystemMessage(self.SYSTEM_PROMPT_TUTOR_AGENT)

        user_id = config["configurable"]["user_id"]

        response = await self.llm_with_tools.ainvoke([system_message] + state["messages"])

        return {"messages": [response]}
    
    # Edges
    def tools_condition(
        self,
        state: Union[list[AnyMessage], dict[str, Any], BaseModel],
        messages_key: str = "messages",
    ) -> Literal["tools", "empty_node"]:
        """Use in the conditional_edge to route to the ToolNode if the last message

        has tool calls. Otherwise, route to the empty node.
        """
        if isinstance(state, list):
            ai_message = state[-1]
        elif isinstance(state, dict) and (messages := state.get(messages_key, [])):
            ai_message = messages[-1]
        elif messages := getattr(state, messages_key, []):
            ai_message = messages[-1]
        else:
            raise ValueError(f"No messages found in input state to tool_edge: {state}")
        if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
            return "tools"
        return "empty_node"

    def build_graph(
        self
    ):
        builder = StateGraph(MessagesState)

        # Nodes
        builder.add_node("empty_node", self.empty_node)
        builder.add_node("memory_agent", self.memory_agent.graph)
        builder.add_node("assistant", self.assistant)
        builder.add_node("tools", ToolNode(self.tools))

        # Edges
        builder.add_edge(START, "memory_agent")
        builder.add_edge("memory_agent", "assistant")
        builder.add_conditional_edges(
            "assistant",
            # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
            # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
            self.tools_condition
        )
        builder.add_edge("tools", "assistant")
        builder.add_edge("empty_node", "memory_agent")

        return builder.compile(
            checkpointer = self.checkpointer,
            store = self.across_thread_memory,
            interrupt_after = ["empty_node"]
        )
    
    def display_graph(self):
        image_data = self.graph.get_graph(xray = 3).draw_mermaid_png()

        with open("graph_image.png", "wb") as f:
            f.write(image_data)

    
