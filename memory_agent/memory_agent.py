import os
from typing import List, Optional
from pydantic import BaseModel, Field

import datetime

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph import MessagesState
from langchain_core.runnables.config import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.base import BaseStore
from langchain_core.messages import merge_message_runs, SystemMessage, HumanMessage
from langgraph.store.memory import InMemoryStore
from trustcall import create_extractor

import memory_agent.memory_agent_prompts as prompts

class UserProfile(BaseModel):
    """User prifile schema with typed fields"""

    name: Optional[str] = Field(description = "The user's name", default = None)
    grade: Optional[str] = Field(description = "Grade of the user", default = None)
    struggles: List[str] = Field(
        description = "Academic struggles of the user, such us math, divisions, history ...",
        default_factory = list
    )
    preferences: List[str] = Field(
        description = "Preferences of the user, for example speaking in spanish",
        default_factory = list
    )
    interests: List[str] = Field(
        description = "Interests that the user has",
        default_factory = list
    )

class MemoryAgent:

    SYSTEM_PROMPT_MEMORY_AGENT = prompts.SYSTEM_PROMPT_MEMORY_AGENT
    TRUSTCALL_INSTRUCTION = prompts.TRUSTCALL_INSTRUCTION


    def __init__(
        self
    ):
        # Define the model to be used
        self.llm = ChatOpenAI(model = "gpt-4o-mini", api_key = os.getenv("OPENAI_API_KEY"), temperature = 0)

        # Define the model with structured output
        self.llm_user_profile = self.llm.with_structured_output(UserProfile)

        # Generate an extractor
        self.extractor = create_extractor(
            self.llm,
            tools = [UserProfile],
            tool_choice = "UserProfile"
        )

        self.graph = self.build_graph()

    # Nodes
    async def update_profile(self, state: MessagesState, config: RunnableConfig, store: BaseStore):
        """Reflect on the chat history and update the memory collection"""

        user_id = config["configurable"]["user_id"]

        # Define the namespace fot the memories
        namespace = ("profile", user_id)

        # Retreive the most recent memories for context
        existing_items = store.search(namespace)

        # Format the existing memories for the Trustcall extractor
        tool_name = "Profile"
        existing_memories = ([(existing_item.key, tool_name, existing_item.value)
                                for existing_item in existing_items]
                                if existing_items
                                else None)
        
        # Merge the chat history and the instruction
        trustcall_instruction = self.TRUSTCALL_INSTRUCTION.format(time = datetime.now().isoformat())

    def build_graph(
        self
    ):
        builder = StateGraph(MessagesState)

        # Nodes
        builder.add_node("call_model", self.call_model)
        builder.add_node("write_memory", self.write_memory)

        # Edges
        builder.add_edge(START, "call_model")
        builder.add_edge("call_model", "write_memory")
        builder.add_edge("write_memory", END)

        checkpointer = MemorySaver()
        self.in_memory_store = InMemoryStore()

        return builder.compile(
            checkpointer = checkpointer,
            store = self.in_memory_store
        )
