#create a python function to runnable

from  langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts  import PromptTemplate
from langchain_core.runnables  import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

def word_counter(text):
    return len(text.split())

model = ChatGoogleGenerativeAI(
    model = "gemini-3.5-flash-lite"
)

parser = StrOutputParser()

template1 = PromptTemplate(
    template = 'Create a joke  on the topic {topic}',
    input_variables = ['topic']
)

chain1 = RunnableSequence(template1, model, parser)

parallel_chain = RunnableParallel({
    'joke' : RunnablePassthrough(),
    'words' : RunnableLambda(word_counter)
})

final_chain = RunnableSequence(chain1, parallel_chain)

response = final_chain.invoke({'topic' : 'teachers'})

print(response['joke'])
print(response['words'])