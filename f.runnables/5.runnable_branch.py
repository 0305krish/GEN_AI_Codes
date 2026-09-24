#create a python function to runnable

from  langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts  import PromptTemplate
from langchain_core.runnables  import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda, RunnableBranch
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

def word_counter(text):
    return len(text.split())

model = ChatGoogleGenerativeAI(
    model = "gemini-3.5-flash-lite"
)

parser = StrOutputParser()

template1 = PromptTemplate(
    template = 'create a report on the topic \n {topic}',
    input_variables = ['topic']
)

template2 = PromptTemplate(
    template = 'create a summary of  the text \n {text}',
    input_variables = ['text']
)

report_gen_chain = RunnableSequence(template1, model, parser)

branch_chain = RunnableBranch(
    (
        RunnableLambda(lambda x : word_counter(x) > 500),
        RunnableSequence(template2, model, parser)
    ),
    RunnablePassthrough()
)

chain = report_gen_chain | branch_chain

response = chain.invoke({'topic' : 'education'})
