from backend import config
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain


def get_doc_types(vectorstore):
    """Extract unique doc_type values from vectorstore metadata."""
    try:
        all_metas = vectorstore._collection.get(include=["metadatas"])["metadatas"]
        doc_types = {m.get("doc_type", "").lower() for m in all_metas if m.get("doc_type")}
        return list(doc_types) or ["general"]
    except Exception as e:
        print(f"Could not extract doc_types: {e}")
        return ["general"]


def build_intent_classifier(doc_types):
    """Build an LLM chain that classifies questions into available doc_types."""
    llm = ChatOpenAI(
        base_url=config.llama_base_url,
        api_key="ollama",
        model="llama3.2:latest",
        temperature=0.0,
    )

    intent_prompt = PromptTemplate.from_template("""
You are an intent classification assistant for Oilwell Corporation.

Classify the user's question into ONE of the following categories:
{doc_types}

If no category applies, return "general".
Output only the category name (lowercase), nothing else.

Question: {question}
Answer:
""")

    return LLMChain(llm=llm, prompt=intent_prompt)


def detect_intent(question: str, intent_chain: LLMChain, doc_types: list) -> str:
    """Run the LLM intent classifier to determine query category."""
    try:
        doc_types_str = ", ".join(doc_types)
        intent = intent_chain.invoke({"question": question, "doc_types": doc_types_str})["text"].strip().lower()
        if intent not in doc_types and intent != "general":
            print(f"Unrecognized intent '{intent}', defaulting to general.")
            intent = "general"
        return intent
    except Exception as e:
        print(f"Intent detection failed: {e}")
        return "general"
