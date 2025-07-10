import os

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph import MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables.config import RunnableConfig
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore



from memory_agent.memory_agent import MemoryAgent
import tutor_agent.tutor_agent_prompts as prompts


class TutorAgent:

    SYSTEM_PROMPT_TUTOR_AGENT = prompts.SYSTEM_PROMPT_TUTOR_AGENT


    def __init__(
        self
    ):
        # Define the model to be used
        self.llm = ChatOpenAI(model = "gpt-4o-mini", api_key = os.getenv("OPENAI_API_KEY"))

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

        response = await self.llm.ainvoke([system_message] + state["messages"])

        return {"messages": [response]}

    def build_graph(
        self
    ):
        builder = StateGraph(MessagesState)

        # Nodes
        builder.add_node("empty_node", self.empty_node)
        builder.add_node("memory_agent", self.memory_agent.graph)
        builder.add_node("assistant", self.assistant)

        # Edges
        builder.add_edge(START, "memory_agent")
        builder.add_edge("memory_agent", "assistant")
        builder.add_edge("assistant", "memory_agent")

        return builder.compile(
            checkpointer = self.checkpointer,
            store = self.across_thread_memory,
            interrupt_after = ["assistant"]
        )

    
