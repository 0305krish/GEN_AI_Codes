from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage,  HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(
    model = "gemini-3.8-flash"
)

chat_history = [
    SystemMessage(content = 'You are a helpful AI_ASSISTANT.')
]

while True:
    user_input = input("You : ")
    chat_history.append(
        HumanMessage(content = user_input)
    )
    if user_input == "exit":
        break;
    response = model.invoke(chat_history)
    chat_history.append(
        AIMessage(content = response.content)
    )
    print("AI : ", response.content)

print(chat_history)