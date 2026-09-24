# RunnableSequence === | (pipe operator)

from  langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts  import PromptTemplate
from langchain_core.runnables  import RunnableSequence
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatGoogleGenerativeAI(
    model = "gemini-3.5-flash-lite"
)

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template = 'Generate a detailed report about {topic}',
    input_variables = ['topic']
)

prompt2 = PromptTemplate(
    template = 'Extract 5 most important points from the given text \n {text}',
    input_variables = ['text']
)

chain = RunnableSequence(prompt1, model, parser, prompt2, model, parser)
response = chain.invoke({'topic' : 'Black Hole'})

print(response)
