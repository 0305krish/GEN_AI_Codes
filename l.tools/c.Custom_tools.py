# need to call own api
# encapsulate business logic
# llm to interact with personal database api etc

from langchain_core.tools import tool

# Step-1 -- Create a function

def multiplty(a, b):
    """Multiplt 2 numbers""" #highly recommended to add this
    return a * b

# Step-3 -- Type Hinting

def multiplty(a : int, b : int) -> int:
    """Multiplt 2 numbers""" #highly recommended to add this
    return a * b

# Step-3 -- Tool Decorator

@tool
def multiplty(a : int, b : int) -> int:
    """Multiplt 2 numbers""" #highly recommended to add this
    return a * b


result = multiplty.invoke({'a' : 3, 'b' : 5})

print(result)

#can also make tools using puydantic model and baseTool class
#can also  club many tools in a tool-kit
