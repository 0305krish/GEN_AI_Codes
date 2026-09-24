from langchain_google_genai import GoogleGenerativeAIEmbeddings
import numpy as np
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

embedding = GoogleGenerativeAIEmbeddings(
    model = "gemini-embedding-2", 
    dimensions = 300
)

document = [
    "Virat Kohli is a great batsman and leader.",
    "Rohit Sharma as Shana is a funny and powerful cricketer",
    "Krishna agarwal is me and i study in MNNIT allahbaad",
    "Swati Agarwal is a teacgher and my mom and very cute."
]

query = "Tell me about virat kohli?"

doc_embeddings = embedding.embed_documents(document)
query_embedding = embedding.embed_query(query)

scores = cosine_similarity([query_embedding], doc_embeddings)[0]

index, scores = sorted(list(enumerate(scores)), key = lambda x : x[1])[-1]

print(query)
print(document[index])




