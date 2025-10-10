import config
import gradio as gr
from backend.RAG_helper.embedding import VectorEmbedding
from backend.RAG_helper.prompt_manager import get_prompts
from backend.RAG_helper.intent_classifier import (
    get_doc_types,
    build_intent_classifier,
    detect_intent
)
from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import HumanMessage, AIMessage
from langchain.callbacks import StdOutCallbackHandler
from langchain.prompts import PromptTemplate
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


# --- Function to initialize system once ---
def initialize_chatbot():
    """Initialize and return all persistent chatbot components."""
    print("🔧 Initializing chatbot components...")

    # Vectorstore
    vectorstore = VectorEmbedding().load_vector()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Dynamically extract document categories from metadata
    doc_types = get_doc_types(vectorstore)
    print(f"Found doc_type categories: {doc_types}")

    # Build LLM intent classifier using those doc_types
    intent_chain = build_intent_classifier(doc_types)

    # LLM
    llm = ChatOpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama",
        model=config.MODEL,
        temperature=0.7,
    )

    # Memory
    chat_history = ChatMessageHistory()
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        chat_memory=chat_history,
        return_messages=True,
        output_key="answer"
    )

    # Prompt templates
    prompts = get_prompts()

    print("Chatbot initialized successfully!")

    return {
        "retriever": retriever,
        "llm": llm,
        "memory": memory,
        "prompts": prompts,
        "intent_chain": intent_chain,
        "doc_types": doc_types
    }


# --- Cache initialization (so it runs only once) ---
_chatbot_components = None


def get_chatbot_components():
    global _chatbot_components
    if _chatbot_components is None:
        _chatbot_components = initialize_chatbot()
    return _chatbot_components


# --- Chat handler ---
def chat(message, history):
    """
    Handle chat with LangChain + Gradio integration.
    Gradio format: [{"role": "user", "content": ...}, {"role": "assistant", "content": ...}]
    """
    components = get_chatbot_components()

    retriever = components["retriever"]
    llm = components["llm"]
    memory = components["memory"]
    prompts = components["prompts"]
    intent_chain = components["intent_chain"]
    doc_types = components["doc_types"]

    # Sync Gradio history with LangChain memory
    memory.chat_memory.clear()
    if history:
        for msg in history:
            if msg["role"] == "user":
                memory.chat_memory.add_message(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                memory.chat_memory.add_message(AIMessage(content=msg["content"]))

    # --- Detect user intent dynamically ---
    intent = detect_intent(message, intent_chain, doc_types)
    print(f"Detected intent: {intent}")

    # --- Pick appropriate prompt ---
    prompt = prompts.get(intent, prompts["general"])

    # --- Build a Conversational Retrieval Chain ---
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt},
        callbacks=[StdOutCallbackHandler()],
        return_source_documents=True
    )

    # --- Generate answer ---
    result = chain.invoke({"question": message})
    return result["answer"]


# --- Gradio launch ---
def gradio_view():
    demo = gr.ChatInterface(
        fn=chat,
        type="messages",
        title="🛢️ Oilwell Corporation Chatbot",
        description="Ask questions about Oilwell's people, products, and documentation",
    )
    demo.launch(inbrowser=True)


if __name__ == "__main__":
    gradio_view()

