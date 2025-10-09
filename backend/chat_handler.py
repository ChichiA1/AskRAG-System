import config
import gradio as gr
from backend.RAG_helper.embedding import VectorEmbedding
from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import HumanMessage, AIMessage
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


# --- Function to initialize system once ---
def initialize_chatbot():
    """Initialize and return a persistent conversational retrieval chain."""
    print(" Initializing chatbot components...")

    # Vectorstore (persistent)
    vectorstore = VectorEmbedding().load_vector()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # LLM (persistent)
    llm = ChatOpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama",  # Dummy key (Ollama ignores it)
        model=config.MODEL,
        temperature=0.7,
    )

    # Memory (persistent)
    chat_history = ChatMessageHistory()
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        chat_memory=chat_history,
        return_messages=True,  # Changed from output_messages
        output_key="answer"
    )

    # Chain (persistent)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True
    )

    print("✓ Chatbot initialized successfully!")
    return conversation_chain


# --- Cached initialization, so it only runs once ---
_chatbot_chain = None


def get_chatbot_chain():
    global _chatbot_chain
    if _chatbot_chain is None:
        _chatbot_chain = initialize_chatbot()
    return _chatbot_chain


# --- Chat handler ---
def chat(message, history):
    """
    Handle chat with proper history format.
    Gradio history format (messages): [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]
    """
    chain = get_chatbot_chain()

    # Sync Gradio history with LangChain memory
    # Clear existing memory and rebuild from Gradio history
    chain.memory.chat_memory.clear()

    if history:
        for msg in history:
            if msg["role"] == "user":
                chain.memory.chat_memory.add_message(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                chain.memory.chat_memory.add_message(AIMessage(content=msg["content"]))

    # Invoke chain with new message
    result = chain.invoke({"question": message})

    return result["answer"]


# --- Gradio launch ---
def gradio_view():
    demo = gr.ChatInterface(
        fn=chat,
        type="messages",
        title="🛢️ Oilwell Corporation Chatbot",
        description="Ask questions about Oilwell's people, products and documentation"
    )
    demo.launch(inbrowser=True)


if __name__ == "__main__":
    gradio_view()
