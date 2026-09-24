from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

chat_template = ChatPromptTemplate([
    ("system", "You are a helpful {domain} expert."),
    MessagesPlaceholder(variable_name='chat_history'),
    ("human", "{query}")
])

chat_history = [] #this will contain information of previous chat

prompt = chat_template.invoke({
    'domain' : 'doctor',
    'chat_history' : chat_history,
    'query' : "what is paracetamol?" 
})

# create-model -> invoke model -> store result