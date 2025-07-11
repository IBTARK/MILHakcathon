import os, requests
from pydantic import BaseModel
from typing import Any, Literal, Union, List

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
import tutor_agent.tutot_agent_prompts as prompts
from langgraph.prebuilt import tools_condition
from langchain_openai import AzureChatOpenAI
from tutor_agent.tools import RAGRetrieveChunks


class TutorState(MessagesState):
    # For the socratic agent
    attempts: int = 0
    max_attempts: int = 3
    query: str = ""
    next: str = ""
    is_solved: bool = False
    socratic_conversation: List[str]

class TutorAgent:

    SYSTEM_PROMPT_SOCRATIC_QUESTIONER = prompts.SYSTEM_PROMPT_SOCRATIC_QUESTIONER
    SYSTEM_PROMPT_ANSWER_EVALUATOR = prompts.SYSTEM_PROMPT_ANSWER_EVALUATOR
    SYSTEM_PROMPT_FINAL_ANSWER = prompts.SYSTEM_PROMPT_FINAL_ANSWER
    SYSTEM_PROMPT_CONGRATULATE = prompts.SYSTEM_PROMPT_CONGRATULATE
    STUDENT_CHECK_SYSTEM_PROMPT = prompts.STUDENT_CHECK_SYSTEM_PROMPT

    def __init__(
        self
    ):
        
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
    def init_state(
        self,
        state: TutorState
    ):
        return {"is_solved": False, "attempts": 0, "max_attempts": 3, "query": state["messages"][-1], "socratic_conversation": [state["messages"][-1]]}

    def router(
        self,
        state: TutorState
    ):
        is_solved = state["is_solved"]
        attempts = state["attempts"]
        max_attempts = state["max_attempts"]

        if not is_solved and attempts < max_attempts:
            return {"next": "socratic_node", "attempts": state["attempts"] + 1}
        elif not is_solved and attempts >= max_attempts:
            return {"next": "final_answer", "attempts": state["attempts"] + 1}
        else:
            return {"next": "congratulate", "attempts": state["attempts"] + 1}

    async def socratic_node(
        self,
        state: TutorState,
        config: RunnableConfig, 
        store: BaseStore
    ):
        user_id = config["configurable"]["user_id"]
        profile_txt = "\n".join(f"{memory.value}" for memory in self.across_thread_memory.search(("profile", user_id)))


        query = state["query"]
        attempts = state["attempts"]
        socratic_interaction = state["messages"][-(attempts * 2):]

        # Format the socratic interaction
        socratic_interaction_txt = "\n".join(f"{msg.type}: {msg.content}" for msg in socratic_interaction)

        system_message = SystemMessage(self.SYSTEM_PROMPT_SOCRATIC_QUESTIONER.format(profile = profile_txt, initial_question = query, history = socratic_interaction_txt))

        response = await self.llm_with_tools.ainvoke([system_message])

        return {"messages": [response], "socratic_conversation": state["socratic_conversation"] + [response]}
    
    async def evaluate_answer(
        self,
        state: TutorState,
        config: RunnableConfig, 
        store: BaseStore
    ):
        user_id = config["configurable"]["user_id"]
        profile_txt = "\n".join(f"{memory}" for memory in self.across_thread_memory.search(("profile", user_id)))

        query = state["query"]
        last_response = state["messages"][-1]

        # Format the socratic interaction
        last_response_txt = f"{last_response.type}: {last_response.content}" 

        system_message = SystemMessage(self.SYSTEM_PROMPT_ANSWER_EVALUATOR.format(profile = profile_txt, initial_question = query, user_answer = last_response_txt))

        response = await self.llm_with_tools.ainvoke([system_message])

        if response.content.lower() == "yes":
            return {"is_solved": True, "socratic_conversation": state["socratic_conversation"] + [response]}
        else:
            return {"is_solved": False, "socratic_conversation": state["socratic_conversation"] + [response]}
        
    async def final_answer(
        self,
        state: TutorState,
        config: RunnableConfig, 
        store: BaseStore
    ):
        user_id = config["configurable"]["user_id"]
        profile_txt = "\n".join(f"{memory}" for memory in self.across_thread_memory.search(("profile", user_id)))

        query = state["query"]
        attempts = state["attempts"]
        socratic_interaction = state["messages"]

        # Format the socratic interaction
        socratic_interaction_txt = "\n".join(f"{msg.type}: {msg.content}" for msg in socratic_interaction if msg.type != "tool_call")

        system_message = SystemMessage(self.SYSTEM_PROMPT_FINAL_ANSWER.format(profile = profile_txt, initial_question = query, history = socratic_interaction_txt))

        response = await self.llm_with_tools.ainvoke([system_message])

        return {"messages": [response], "attempts": 0, "next": "", "is_solved": False, "socratic_conversation": state["socratic_conversation"] + [response]}
    
    async def congratulate(
        self,
        state: TutorState,
        config: RunnableConfig, 
        store: BaseStore
    ):
        user_id = config["configurable"]["user_id"]
        profile_txt = "\n".join(f"{memory}" for memory in self.across_thread_memory.search(("profile", user_id)))

        query = state["query"]
        attempts = state["attempts"]
        socratic_interaction = state["messages"][-(attempts * 2):]

        # Format the socratic interaction
        socratic_interaction_txt = "\n".join(f"{msg.type}: {msg.content}" for msg in socratic_interaction)

        system_message = SystemMessage(self.SYSTEM_PROMPT_CONGRATULATE.format(profile = profile_txt, initial_question = query, history = socratic_interaction_txt))

        response = await self.llm_with_tools.ainvoke([system_message])

        return {"messages": [response], "attempts": 0, "next": "", "is_solved": False, "socratic_conversation": state["socratic_conversation"] + [response]}
    
    async def student_check(
        self,
        state: TutorState,
        config: RunnableConfig, 
        store: BaseStore
    ):

        user_id = config["configurable"]["user_id"]
        profile_txt = "\n".join(f"{memory}" for memory in self.across_thread_memory.search(("profile", user_id)))

        query = state["query"]
        attempts = state["attempts"]
        max_attempts = state["max_attempts"]
        socratic_interaction = state["socratic_conversation"]

        history = "\n".join(f"role: {msg.type} content: {msg.content}" for msg in socratic_interaction)
        
        TOKEN   = os.getenv("TG_BOT_TOKEN")
        CHAT_ID = os.getenv("TG_CHAT_ID")
        url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"

        system_prompt = SystemMessage(content = self.STUDENT_CHECK_SYSTEM_PROMPT.format(user_profile = profile_txt, attempts = attempts, max_attempts = max_attempts, history = history))
        result = await self.llm.ainvoke([system_prompt])

        requests.post(url, data={"chat_id": CHAT_ID, "text": result.content}, timeout=5)

        return None
    
    # Edges
    def determine_action(
        self, 
        state: TutorState
    ) -> Literal["socratic_node", "final_answer", "congratulate"]:
        next = state["next"]

        return next

    def socratic_tools_condition(
        self,
        state: Union[list[AnyMessage], dict[str, Any], BaseModel],
        messages_key: str = "messages",
    ) -> Literal["socratic_tools", "evaluate_answer"]:
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
            return "socratic_tools"
        return "evaluate_answer"

    def final_answer_tools_condition(
        self,
        state: Union[list[AnyMessage], dict[str, Any], BaseModel],
        messages_key: str = "messages",
    ) -> Literal["final_answer_tools", "student_check"]:
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
            return "final_answer_tools"
        return "student_check"
    
    def evaluate_answer_tools_condition(
        self,
        state: Union[list[AnyMessage], dict[str, Any], BaseModel],
        messages_key: str = "messages",
    ) -> Literal["evaluate_answer_tools", "router"]:
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
            return "evaluate_answer_tools"
        return "router"


    def build_graph(self):
        builder = StateGraph(TutorState)

        # Nodes
        builder.add_node("memory_agent", self.memory_agent.graph)
        builder.add_node("init_state", self.init_state)
        builder.add_node("router", self.router)
        builder.add_node("socratic_node", self.socratic_node)
        builder.add_node("evaluate_answer", self.evaluate_answer)
        builder.add_node("final_answer", self.final_answer)
        builder.add_node("congratulate", self.congratulate)
        builder.add_node("student_check", self.student_check)
        builder.add_node("socratic_tools", ToolNode(self.tools))
        builder.add_node("final_answer_tools", ToolNode(self.tools))
        builder.add_node("evaluate_answer_tools", ToolNode(self.tools))

        # Edges
        builder.add_edge(START, "memory_agent")
        builder.add_edge("memory_agent", "init_state")
        builder.add_edge("init_state", "router")
        builder.add_conditional_edges(
            "router",
            self.determine_action
        )

        builder.add_conditional_edges(
            "socratic_node",
            # If the latest message (result) from socratic_node is a tool call -> tools_condition routes to tools
            # If the latest message (result) from socratic_node is a not a tool call -> tools_condition routes to END
            self.socratic_tools_condition,
        )
        builder.add_edge("socratic_tools", "socratic_node")
        builder.add_edge("evaluate_answer", "router")

        builder.add_conditional_edges(
            "evaluate_answer",
            self.evaluate_answer_tools_condition
        )

        builder.add_edge("evaluate_answer_tools", "evaluate_answer")

        builder.add_conditional_edges(
            "final_answer",
            self.final_answer_tools_condition
        )
        builder.add_edge("final_answer_tools", "final_answer")
        
        builder.add_edge("congratulate", "student_check")
        builder.add_edge("student_check", "memory_agent")

        return builder.compile(
            checkpointer = self.checkpointer,
            store = self.across_thread_memory,
            interrupt_after = ["socratic_node", "student_check"]
        )
    
    def display_graph(self):
        image_data = self.graph.get_graph(xray = 3).draw_mermaid_png()

        with open("graph_image.png", "wb") as f:
            f.write(image_data)