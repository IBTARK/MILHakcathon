import chainlit as cl
import pathlib
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage

from tutor_agent.tutor_agent import TutorAgent
from utils import ingest_file

load_dotenv()

# Runs once when the chat session starts
@cl.on_chat_start
async def chat_start():
    agent = TutorAgent()
    configuration = {"configurable": {"thread_id": "1", "user_id": "Lance"}}

    cl.user_session.set("agent", agent)
    cl.user_session.set("thread", configuration)
    cl.user_session.set("first_run", True)

    await cl.Message(
        content = "👋 ¡Hola! Soy tu tutor. ¿En qué puedo ayudarte?"
    ).send()

# Executes every time the user sends a message
@cl.on_message
async def handle_message(message: cl.Message):
    user_msg = message.content

    agent = cl.user_session.get("agent")
    configuration = cl.user_session.get("thread")


    # If the message has a file
    if message.elements:
        for f in message.elements:
            temp_path = pathlib.Path(f.path)

            print(f"vecorizing file: {temp_path}")

            await ingest_file(temp_path)
            temp_path.unlink() 


    if cl.user_session.get("first_run"):
        state = await agent.graph.ainvoke({"messages": [HumanMessage(user_msg)]}, configuration)
        cl.user_session.set("first_run", False)
    else:
        agent.graph.update_state(
            configuration,
            {"messages": [HumanMessage(content = user_msg)]}
        )
        state = await agent.graph.ainvoke(None,  configuration)

    # Send the response back to the front-end
    await cl.Message(content = state["messages"][-1].content).send()

