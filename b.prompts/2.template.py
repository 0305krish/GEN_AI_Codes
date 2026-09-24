from langchain_core.prompts import PromptTemplate

# Prompt template
prompt = PromptTemplate(
    input_variables=[
        "paper",
        "output_type",
        "output_length",
        "question"
    ],
    template="""
You are a research assistant.

Research Paper: {paper}

User Question:
{question}

Output Type:
{output_type}

Output Length:
{output_length}

Instructions:
- Answer based on the selected research paper.
- Keep the response accurate and research-oriented.
- Do not make up information.
- Structure the answer according to the requested output type.
- Keep the response within the requested length.

Now generate the answer.
"""
)

prompt.save("template.json")