from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.runnables  import RunnableParallel, RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

model = ChatGoogleGenerativeAI(
    model = "gemini-3.5-flash-lite"
)

class FeedbackResult(BaseModel):
    sentiment : Literal['positive', 'negative']
pydantic_output_parser = PydanticOutputParser(pydantic_object = FeedbackResult)


feedback = """I recently purchased this product and have mixed feelings about it. The build quality is quite good, and the product looks premium and feels durable. It was also easy to set up and use. However, the battery life is disappointing, and the device sometimes becomes slow when multiple applications are running. Considering its price, I expected better performance. Overall, it is a decent product with some good features, but there are several areas that could be improved."""

template = PromptTemplate(
    input_variables = ['feedback'],
    template = 'Tell me the sentiment of the follow ing feedback if it is positive or negative \n {feedback} \n {format_instructions}',
    partial_variables = {'format_instructions' : pydantic_output_parser.get_format_instructions()}
)

prompt1 = PromptTemplate(
    template = 'Write a suitable response for the positive feedback \n {feedback}', 
    input_variables = ['feedback']
)

prompt2 = PromptTemplate(
    template = 'Write a suitable response for the negative feedback \n {feedback}', 
    input_variables = ['feedback']
)

parser = StrOutputParser()

sentiment_chain = template | model | pydantic_output_parser

parallel_chain = RunnableParallel(
    feedback=lambda x: x["feedback"],
    sentiment=sentiment_chain
)

branch_chain = RunnableBranch(
    (lambda x:x.sentiment == 'positive', prompt1 | model | parser),
    (lambda x:x.sentiment == 'negative', prompt2 | model | parser),
    RunnableLambda(lambda x:'could not find sentiment')
)

branch_chain = RunnableBranch(
    (
        lambda x: x["sentiment"].sentiment == "positive",
        prompt1 | model | parser
    ),
    (
        lambda x: x["sentiment"].sentiment == "negative",
        prompt2 | model | parser
    ),
    lambda x: "Could not find sentiment"
)

final_chain = parallel_chain | branch_chain

result = final_chain.invoke({'feedback' : feedback})

print(result)
