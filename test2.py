from dotenv import load_dotenv
import asyncio
from tutor_agent.tutor_agent import TutorAgent
from langchain_core.messages import SystemMessage, HumanMessage


load_dotenv()


agent = TutorAgent()

query = "¿Qué prefijo de red debe anunciar el encaminador vm_3router?"
msg = HumanMessage(content = query)

messages = [HumanMessage(content=query)]
configuration = {"configurable": {"thread_id": "1", "user_id": "Lance"}}
response = asyncio.run(agent.graph.ainvoke({"messages": messages}, configuration))

for m in response["messages"]:
    m.pretty_print()