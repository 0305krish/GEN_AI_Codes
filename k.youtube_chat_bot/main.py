from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
load_dotenv()
model = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    temperature=0.5
)

#loading transcript
video_id = "LPZh9BOjkQs"
api = YouTubeTranscriptApi()
transcript = api.fetch(video_id, languages = ['en'])
text = " ".join(snippet.text for snippet in transcript)

#text split
splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 50
)

chunks = splitter.split_text(text)

documents = [
    Document(page_content=chunk)
    for chunk in chunks
]

#vector store
embeddings = GoogleGenerativeAIEmbeddings(
    model = "gemini-embedding-2", 
    dimensions = 512
)

vectorStore = Chroma.from_documents(
    documents = documents,
    embedding = embeddings,
    persist_directory = './chroma_db'
)

#retrievers
retriever = vectorStore.as_retriever(
    search_type = "similarity",
    search_kwargs = {'k' : 4}
)

#prompt and augmentation

prompt = PromptTemplate(
    template = """You are a helpful AI-ASSISTANT.
    I have given u some context that is as follow :- \n
    {context} \n
    Your work is to answer the question only with the help of the context 
    and if u didnt find a suitable  answer just give i dont know.
    The question is as follows :- \n
    {question}""",
    input_variables = ['context', 'question']
)

question = 'Who is Virat Kohli?'
def format_docs(retrieved_docs):
    context = "\n\n".join(doc.page_content for doc in retrieved_docs)
    return context

#generation

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'question' : RunnablePassthrough(),
    'context' : retriever | RunnableLambda(format_docs)
})

generation_chain = parallel_chain | prompt | model | parser

result = generation_chain.invoke(question)



