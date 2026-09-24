#import -> model creation -> prompt template -> ui
# -> button -> final prompt structure -> display


import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate, load_prompt

load_dotenv()

# LLM
model = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    temperature=0.5
)

prompt = load_prompt('template.json')
# ---------------- UI ----------------

st.title("📚 Research Assistant")

paper = st.selectbox(
    "Select Research Paper",
    [
        "Attention Is All You Need",
        "BERT: Pre-training of Deep Bidirectional Transformers",
        "ResNet: Deep Residual Learning for Image Recognition",
        "Generative Adversarial Networks"
    ]
)

output_type = st.selectbox(
    "Output Type",
    [
        "Summary",
        "Critical Analysis",
        "Literature Review",
        "Key Findings",
        "Methodology",
        "Research Gaps"
    ]
)

output_length = st.selectbox(
    "Output Length",
    [
        "Short",
        "Medium",
        "Detailed"
    ]
)

question = st.text_area(
    "What do you want to know?",
    placeholder="Explain the main contribution of this paper."
)

if st.button("Generate Research"):

    if question:

        #chain = prompt | model
        #response = chain.invoke({
        #    "paper": paper,
        #    "output_type": output_type,
        #    "output_length": output_length,                    "question": question
        #})
        
        final_prompt = prompt.invoke({
            "paper": paper,
            "output_type": output_type,
            "output_length": output_length,
            "question": question
        })

        response = model.invoke(final_prompt)

        st.subheader("Research Output")
        st.write(response.content)

    else:
        st.warning("Please enter a question.")