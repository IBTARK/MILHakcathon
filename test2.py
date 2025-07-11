from dotenv import load_dotenv
import asyncio
from tutor_agent.tutor_agent import TutorAgent
from langchain_core.messages import SystemMessage, HumanMessage


load_dotenv()


agent = TutorAgent()

agent.display_graph()