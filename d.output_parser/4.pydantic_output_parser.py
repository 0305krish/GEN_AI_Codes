#enforce schema validation

from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()

model = ChatGoogleGenerativeAI(
    model = "gemini-3.5-flash-lite"
)

class Person(BaseModel):
    name : str
    age : int = Field(gt = 18, description= 'age of person')
    city : str

parser = PydanticOutputParser(pydantic_object = Person)

template = PromptTemplate(
    template = "generate the name, age, city of a famous indian \n {format_instructions}",
    input_variables = [],
    partial_variables = {'format_instructions' : parser.get_format_instructions()}
)

chain = template | model | parser

final_result = chain.invoke({})

print(final_result)