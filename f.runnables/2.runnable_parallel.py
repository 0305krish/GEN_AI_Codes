#it is important that the input should be same in run parallel to all the chains inside it

from  langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts  import PromptTemplate
from langchain_core.runnables  import RunnableSequence, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatGoogleGenerativeAI(
    model = "gemini-3.5-flash-lite"
)

parser = StrOutputParser()

template1 = PromptTemplate(
    template = 'Create detailed notes out of the given text \n {text}',
    input_variables = ['text']
)

template2 = PromptTemplate(
    template = 'Create a challenging quiz out of the given text \n {text}',
    input_variables = ['text']
)

parallel_chain = RunnableParallel({
    'notes' : RunnableSequence(template1, model, parser),
    'quiz' : RunnableSequence(template2, model, parser)
})

response = parallel_chain.invoke({'text' : 'langchain is a poweful library'})

print(response['notes'])
print(response['quiz'])