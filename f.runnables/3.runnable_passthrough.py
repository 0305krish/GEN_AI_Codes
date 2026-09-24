#What ever input  we give to it the same output will be returned

from  langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts  import PromptTemplate
from langchain_core.runnables  import RunnableSequence, RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatGoogleGenerativeAI(
    model = "gemini-3.5-flash-lite"
)

parser = StrOutputParser()

template1 = PromptTemplate(
    template = 'Create a joke  on the topic {topic}',
    input_variables = ['topic']
)

template2 = PromptTemplate(
    template = 'Give explanation of the joke \n {joke}',
    input_variables = ['joke']
)

chain1 = RunnableSequence(template1, model, parser)

parallel_chain = RunnableParallel({
    'joke' : RunnablePassthrough(),
    'explanation' : RunnableSequence(template2, model, parser)
})

final_chain = RunnableSequence(chain1, parallel_chain)

response = final_chain.invoke({'topic' : 'teachers'})

print(response['joke'])
print(response['explanation'])