import os

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph import MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import SystemMessage, HumanMessage


import tutor_agent.tutor_agent_prompts as prompts


class Tutor:

    SYSTEM_PROMPT_TUTOR_AGENT = prompts.SYSTEM_PROMPT_TUTOR_AGENT


    def __init__(
        self
    ):
        # Define the model to be used
        self.llm = ChatOpenAI(model = "gpt-4o-mini", api_key = os.getenv("OPENAI_API_KEY"))

        self.graph = self.build_graph()

    # Nodes
    def empty_node(
        self,
        state: MessagesState
    ):
        return {}

    async def assistant(
        self,
        state: MessagesState
    ):
        system_message = SystemMessage(self.SYSTEM_PROMPT_TUTOR_AGENT)

        print(f"system_message: {system_message}")

        response = await self.llm.ainvoke([system_message] + state["messages"])

        print(f"response: {response}")

        return {"messages": [response]}

    def build_graph(
        self
    ):
        builder = StateGraph(MessagesState)

        # Nodes
        builder.add_node("empty_node", self.empty_node)
        builder.add_node("assistant", self.assistant)

        # Edges
        builder.add_edge(START, "empty_node")
        builder.add_edge("empty_node", "assistant")
        builder.add_edge("assistant", "assistant")

        checkpointer = MemorySaver()

        return builder.compile(
            checkpointer = checkpointer,
            interrupt_after = ["assistant"]
        )

    
