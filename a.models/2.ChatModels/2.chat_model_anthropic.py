from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

load_dotenv()

model = ChatAnthropic(
    model = "",
    temperature = 0
)

result = model.invoke("How to treat my girl bestie if she is not giving me timely replies?")

print(result)