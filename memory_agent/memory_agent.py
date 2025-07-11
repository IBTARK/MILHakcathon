import os
import uuid
from typing import List, Optional
from pydantic import BaseModel, Field

from datetime import datetime

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph import MessagesState
from langchain_core.runnables.config import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.base import BaseStore
from langchain_core.messages import merge_message_runs, SystemMessage, HumanMessage
from langgraph.store.memory import InMemoryStore
from trustcall import create_extractor
from langchain_openai import AzureChatOpenAI


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
        self,
        checkpointer: MemorySaver,
        across_thread_memory: InMemoryStore
    ):
        # Define the model to be used
        endpoint = "https://hackaton-scv-openai.openai.azure.com/"
        model_name = "gpt-4.1-mini"
        deployment_name = "gpt-4.1-mini-team6"
        
        # Define the model to be used
        self.llm = AzureChatOpenAI(
            azure_endpoint = endpoint,
            azure_deployment = deployment_name,
            api_version = os.getenv("AZURE_API_VERSION"),
            api_key = os.getenv("AZURE_API_KEY"),
            model_name = model_name,
        )

        # Define the model with structured output
        self.llm_user_profile = self.llm.with_structured_output(UserProfile)

        self.checkpointer = checkpointer
        self.across_thread_memory = across_thread_memory

        # Generate an extractor
        self.extractor = create_extractor(
            self.llm,
            tools = [UserProfile],
            tool_choice = "UserProfile"
        )

        self.graph = self.build_graph()

    # Nodes
    def update_profile(self, state: MessagesState, config: RunnableConfig, store: BaseStore):
        """Reflect on the chat history and update the memory collection"""

        user_id = config["configurable"]["user_id"]
        
        # Define the namespace fot the memories
        namespace = ("profile", user_id)

        # Retreive the most recent memories for context
        existing_items = store.search(namespace)

        # Format the existing memories for the Trustcall extractor
        tool_name = UserProfile.__name__ 
        existing_memories = ([(existing_item.key, tool_name, existing_item.value)
                                for existing_item in existing_items]
                                if existing_items
                                else None)
        
        # Merge the chat history and the instruction
        trustcall_instruction = self.TRUSTCALL_INSTRUCTION.format(time = datetime.now().isoformat())
        updated_messages = list(merge_message_runs(messages = [SystemMessage(content = trustcall_instruction)] + state["messages"]))

        for memory in self.across_thread_memory.search(("profile", user_id)):
            print(memory.value)

        # Invoke the extractor
        result = self.extractor.invoke({"messages": updated_messages, 
                                         "existing": existing_memories})
        
        # Save the memories from Trustcall to the store
        for r, rmeta in zip(result["responses"], result["response_metadata"]):
            store.put(namespace,
                    rmeta.get("json_doc_id", str(uuid.uuid4())),
                    r.model_dump(mode = "json"),
                )
            
        for memory in self.across_thread_memory.search(("profile", user_id)):
            print(memory.value)
        
        return {}


    def build_graph(
        self
    ):
        builder = StateGraph(MessagesState)

        # Nodes
        builder.add_node("update_profile", self.update_profile)

        # Edges
        builder.add_edge(START, "update_profile")
        builder.add_edge("update_profile", END)

        return builder.compile(
            store = self.across_thread_memory
        )
