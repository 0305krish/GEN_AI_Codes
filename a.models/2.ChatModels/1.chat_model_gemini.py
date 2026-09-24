from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

model = ChatGoogleGenerativeAI(
    model = "gemini-3.8-flash",
    temperature = 1.6,
)

response = model.invoke("Suggest me places to visit in India")

print(response.content[0]["text"])