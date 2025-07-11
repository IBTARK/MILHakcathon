from dotenv import load_dotenv
import requests, os

from tutor_agent.tutor_agent import TutorAgent

load_dotenv()


TOKEN   = os.getenv("TG_BOT_TOKEN")   # o pega tu token como string
CHAT_ID = os.getenv("TG_CHAT_ID")     # o tu número
TEXT    = "Mensaje urgente"

url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
requests.post(url, data={"chat_id": CHAT_ID, "text": TEXT}, timeout=5)
print("enviado")


agent = TutorAgent()

agent.display_graph()