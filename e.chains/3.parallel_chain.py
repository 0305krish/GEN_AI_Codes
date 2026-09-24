from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.runnables  import RunnableParallel

load_dotenv()

model = ChatGoogleGenerativeAI(
    model = "gemini-3.5-flash-lite"
)

template1 = PromptTemplate(
    template = 'Create detailed notes out of the given text \n {text}',
    input_variables = ['text']
)

template2 = PromptTemplate(
    template = 'Create a challenging quiz out of the given text \n {text}',
    input_variables = ['text']
)

template3 = PromptTemplate(
    template  = "Merge the provided notes and quiz into a single document \n {notes} \n {quiz}",
    input_variables = ['notes', 'quiz']
)

parser = StrOutputParser()

text  = """LangChain is a framework designed to make it easier for developers to build applications powered by large language models. A large language model can generate text, answer questions, summarize documents, extract structured information, write code, and perform many other language-related tasks. However, building a complete application around an LLM usually requires much more than simply sending a prompt to a model. Applications often need prompt templates, multiple model calls, output parsers, memory or message history, external tools, databases, document retrieval, embeddings, vector stores, and custom business logic. LangChain provides abstractions that allow these components to be connected together into reusable pipelines.

One of the most important concepts in LangChain is the idea of a Runnable. A Runnable represents a component that can receive an input and produce an output. Many LangChain components implement this interface, including prompt templates, chat models, output parsers, retrievers, and chains. Because these components follow a common interface, they can be connected together using the pipe operator. For example, a prompt can be connected to a language model, and the model can then be connected to an output parser. This creates a pipeline where the output of one component becomes the input of the next component.

The pipe operator is represented using the vertical bar symbol. A simple LangChain pipeline might look like prompt | model | parser. The prompt first receives the user's input and creates a formatted prompt. The model receives that formatted prompt and generates a response. The parser then processes the model's response into the desired output format. This approach is commonly referred to as LCEL, or LangChain Expression Language. LCEL allows developers to construct chains declaratively instead of manually calling every component one by one.

Prompt templates are another fundamental concept in LangChain. Instead of writing a fixed prompt every time, developers can create a reusable template containing variables. For example, a prompt template might contain the sentence "Explain the following topic in detail: {topic}". When the chain is invoked with a value such as "black holes", LangChain replaces the topic variable with the provided value. This makes prompts reusable and allows the same pipeline to process many different inputs.

There are several types of prompts available in LangChain. PromptTemplate is commonly used for simple text prompts, while ChatPromptTemplate is designed for chat-based models and allows developers to define different message roles such as system, human, and assistant. A system message can define the behavior or role of the model, while a human message contains the user's actual request. Message placeholders can also be used when previous conversation messages need to be inserted dynamically into a prompt.

Language models are responsible for generating the actual response. LangChain supports different model providers, allowing developers to work with models from different companies through a similar programming interface. For example, developers can use models provided by Google, OpenAI, Anthropic, and other providers. A chat model generally receives messages and returns an AI message. The content of that message can then be passed to another component in the chain.

Output parsers are used when the raw output of a model needs to be transformed into a more useful representation. StrOutputParser is one of the simplest parsers and extracts the textual content from a model response. Other parsers can be used when developers want structured outputs such as JSON, lists, or objects. Structured output is particularly useful when an application needs to consume model responses programmatically rather than simply displaying them to a user.

One important concept when building LangChain pipelines is understanding the data type flowing between components. A prompt template may expect a dictionary containing named variables, while a model may return an AIMessage object, and a string parser may return a plain Python string. Developers must understand these input and output types because a chain can fail if one component produces a type that the next component does not expect.

For example, suppose the first prompt generates a detailed report about a topic. After the model generates the report, a string parser converts the result into a string. If a second prompt expects a variable called text, the second prompt cannot necessarily receive the string directly. The string needs to be placed inside a dictionary with the appropriate key. LangChain provides runnable mappings that make this kind of data transformation possible. A common pattern is {"text": first_chain} | second_prompt. In this pattern, the output of first_chain is assigned to the text variable expected by second_prompt.

This concept becomes even more powerful when multiple values need to be passed to a later component. A developer might have one chain generating a report and another value containing the original topic. Both can be combined into a dictionary and passed to the next prompt. This allows complex pipelines to combine information from different sources. RunnablePassthrough can also be used when an existing input needs to be forwarded to another component without modification.

LangChain pipelines can therefore be thought of as data-flow systems. Instead of thinking only about individual function calls, developers should think about what data enters each component, what data leaves it, and what format the next component expects. A successful chain is one in which the output type of each stage is compatible with the input type of the following stage, or where an appropriate transformation is inserted between them.

Another important concept is composition. A chain does not have to contain only one model call. A developer can create a first chain that researches a topic, a second chain that summarizes the research, and a third chain that converts the summary into a structured format. These chains can then be composed into a larger workflow. This makes it possible to build sophisticated applications from relatively small reusable components.

LangChain is also commonly used for retrieval-augmented generation, often called RAG. In a RAG system, information is retrieved from an external knowledge source before the language model generates an answer. This is useful because language models may not have access to a company's private documents, a specific collection of files, or information that has changed after the model was trained. A retrieval system can search relevant documents and provide their contents to the model as context.

Embeddings are an important part of many retrieval systems. An embedding model converts text into numerical vectors that represent semantic information. Texts with similar meanings tend to have vectors that are close together in vector space. These vectors can be stored in vector databases or vector stores. When a user asks a question, the question can also be converted into an embedding, and the system can search for documents whose embeddings are semantically similar to the question.

A typical RAG pipeline contains several stages. Documents are first loaded from sources such as PDFs, websites, text files, or databases. Large documents are usually divided into smaller chunks using a text splitter. Each chunk is converted into an embedding and stored in a vector store. When a user asks a question, the question is embedded and used to retrieve relevant chunks. The retrieved chunks are then inserted into a prompt and sent to the language model. The model uses the retrieved information to generate the final response.

Document chunking is important because language models have limits on how much information they can process in a single request. If a document is extremely large, sending the entire document every time may be inefficient or impossible. Splitting the document into smaller chunks allows the retrieval system to select only the relevant pieces. However, chunk size and overlap must be chosen carefully. Very small chunks may lose context, while extremely large chunks may contain unnecessary information.

LangChain can also interact with external tools. A tool is a function that a model can use to perform an action or retrieve information. Examples include web search, database queries, calculators, APIs, file operations, and custom Python functions. Tool calling allows a language model to decide when additional information or computation is necessary instead of relying entirely on its internal knowledge.

Agents build upon this idea by allowing a model to choose between tools dynamically. Instead of following one fixed pipeline, an agent can reason about the user's request and determine which tools should be used. For example, an application could provide an agent with a calculator, a database search tool, and a web search tool. Depending on the question, the agent could select the appropriate tool and use its result to formulate an answer.

However, fixed chains and agents serve different purposes. A fixed chain is generally easier to understand and control because the sequence of operations is predetermined. Agents provide more flexibility because the model can select actions dynamically, but this can also make behavior more difficult to predict and debug. Developers should therefore choose the architecture based on the requirements of the application rather than automatically using agents for every task.

Memory and conversation history are another important area of LLM application development. A chatbot may need to remember previous messages so that later questions can be interpreted in context. LangChain provides mechanisms for working with message history and conversation state. Instead of sending only the latest user message, the application can provide relevant previous messages to the model. This allows the model to maintain conversational context.

Modern LangChain applications often separate prompts, models, parsers, retrieval components, and application logic into distinct pieces. This modular design makes applications easier to test and maintain. For example, a developer can change the model provider without rewriting the entire application, or replace one output parser with another while keeping the rest of the pipeline unchanged.

Debugging LangChain applications requires paying close attention to intermediate outputs. If a large chain fails, developers should break it into smaller chains and inspect the output of each stage. They should check whether a prompt received the expected variables, whether the model returned the expected message type, whether the parser successfully processed the model output, and whether the next component received the correct structure.

A useful mental model is to treat a LangChain application like a series of connected functions. Each component has an input contract and an output contract. Prompt templates transform structured input variables into prompts. Models transform prompts into model responses. Parsers transform model responses into application-friendly values. Runnable mappings transform or combine data so that downstream components receive the variables they require.

The ability to compose these components is one of the central strengths of LangChain. A simple application may consist of only a prompt, a model, and a parser, while a more advanced application can contain retrieval, multiple prompts, multiple models, tools, structured outputs, memory, and external APIs. Despite this complexity, the same basic principle remains important: understand the data flowing through every stage and explicitly transform that data whenever the next component expects a different structure.

For someone learning LangChain, it is therefore useful to first understand basic LCEL composition, PromptTemplate, ChatPromptTemplate, model invocation, StrOutputParser, RunnablePassthrough, RunnableLambda, and dictionary-based Runnable mappings. After understanding these concepts, it becomes easier to learn structured output, parallel execution, branching, retrieval chains, RAG, tool calling, agents, and more advanced application architectures.

Ultimately, LangChain is not simply a library for calling an LLM. It provides a framework for connecting language models with prompts, data sources, tools, parsers, retrieval systems, and application logic. The most important skill when learning it is not memorizing every class or method, but understanding how components communicate with one another and how information can be transformed as it moves through a pipeline."""

# chain1 = template1 | model | parser
# chain2 = template2 | model | parser
# chain3 = template3 | model | parser

# response1 = chain1.invoke({'text' : text})
# response2 = chain2.invoke({'text' : text})

# result = chain3.invoke({
#     'notes' : response1,
#     'quiz' : response2
# })

parallel_chain = RunnableParallel({
    'notes' : template1 | model | parser,
    'quiz' : template2 | model | parser
})

merge_chain = template3 | model | parser
chain = parallel_chain | merge_chain

result = chain.invoke({'text' : text})

print(result)

