from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = GoogleGenerativeAIEmbeddings(
    model = "gemini-embedding-2", 
    dimensions = 32
)

result = embedding.embed_query("My name is Krishna Agarwal")

print(str[result])