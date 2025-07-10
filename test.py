import os
from dotenv import load_dotenv
from typing import TypedDict, List
import asyncio

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph import MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.store.memory import InMemoryStore
from trustcall import create_extractor

from memory_agent.memory_agent import MemoryAgent

load_dotenv()

agent = MemoryAgent()

config = {"configurable": {"thread_id": "1", "user_id": "1"}}

# User input 
input_messages = [HumanMessage(content="Hi, my name is Lance")]

print("First\n")

state = asyncio.run(agent.graph.ainvoke({"messages": input_messages}, config))
state["messages"][-1].pretty_print()


print("Second\n")

# User input 
input_messages = [HumanMessage(content="I like to bike around San Francisco")]

# Run the graph
state = asyncio.run(agent.graph.ainvoke({"messages": input_messages}, config))
state["messages"][-1].pretty_print()



# Namespace for the memory to save
user_id = "1"
namespace = ("memory", user_id)
existing_memory = agent.in_memory_store.get(namespace, "user_memory")
existing_memory.dict()


print(existing_memory.value)


