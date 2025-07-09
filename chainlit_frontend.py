import chainlit as cl
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage

from tutor_agent.tutor_agent import Tutor


load_dotenv()

# Runs once when the chat session starts
@cl.on_chat_start
async def chat_start():
    agent = Tutor()
    configuration = {"configurable": {"thread_id": "1"}}

    cl.user_session.set("agent", agent)
    cl.user_session.set("thread", configuration)

    await cl.Message(
        content = "👋 ¡Hola! Soy tu tutor. ¿En qué puedo ayudarte?"
    ).send()

# Executes every time the user sends a message
@cl.on_message
async def handle_message(message: cl.Message):
    user_msg = message.content

    agent = cl.user_session.get("agent")
    configuration = cl.user_session.get("thread")

    print("Hola")

    state = await agent.graph.ainvoke({"messages": [HumanMessage(user_msg)]}, configuration)

    print(state["messages"])

    # Send the response back to the front-end
    await cl.Message(content = state["messages"][-1].content).send()

