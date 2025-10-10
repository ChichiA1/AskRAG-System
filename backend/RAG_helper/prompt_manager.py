from langchain.prompts import PromptTemplate


def get_prompts():
    """Return prompt templates for each category."""
    return {
        "policy": PromptTemplate.from_template("""
You are a compliance assistant for Oilwell Corporation.
Provide clear, accurate, and policy-compliant answers.
Quote directly from retrieved context where relevant.

Context:
{context}

Chat History:
{chat_history}

Question:
{question}

Answer:
"""),

        "employee": PromptTemplate.from_template("""
You are a friendly HR assistant for Oilwell Corporation.
Answer questions about employees professionally and politely.

Context:
{context}

Chat History:
{chat_history}

Question:
{question}

Answer:
"""),

        "product": PromptTemplate.from_template("""
You are a technical documentation specialist for Oilwell Corporation.
Answer precisely with relevant specs, features, and use cases.

Context:
{context}

Chat History:
{chat_history}

Question:
{question}

Answer:
"""),

        "general": PromptTemplate.from_template("""
You are a helpful general-purpose assistant for Oilwell Corporation.
Answer clearly and concisely based on the provided context.

Context:
{context}

Chat History:
{chat_history}

Question:
{question}

Answer:
"""),
    }
