#model -> parser -> template -> final-parsing
#we cant decide schema through json object parser
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

model = ChatGoogleGenerativeAI(
    model = "gemini-3.5-flash-lite"
)

parser = JsonOutputParser()

template = PromptTemplate(
    template = "Write down the name, age and city of a fictional person \n {format_instruction}",
    input_variables = [],
    partial_variables = {'format_instruction' : parser.get_format_instructions()}
)

# prompt = template.format()

# result = model.invoke(prompt)

# final_result = parser.parse(result.content[0]["text"])

chain = template | model | parser

final_result = chain.invoke({})

print(final_result)
