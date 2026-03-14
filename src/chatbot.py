import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

from src.config import CHAT_MODEL, GEMINI_API_KEY


genai.configure(api_key=GEMINI_API_KEY)


def create_chatbot(vectorstore):

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    llm = ChatGoogleGenerativeAI(
        model=CHAT_MODEL,
        temperature=0.2,
        google_api_key=GEMINI_API_KEY
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        output_key="answer"
    )

    return qa_chain


def create_llm():
    """Create a direct LLM response function without vector retrieval."""
    llm = ChatGoogleGenerativeAI(
        model=CHAT_MODEL,
        temperature=0.3,
        google_api_key=GEMINI_API_KEY
    )

    def llm_query(prompt_text: str) -> str:
        try:
            return llm.predict(prompt_text)
        except Exception as e:
            try:
                model = genai.GenerativeModel(CHAT_MODEL)
                response = model.generate_content(
                    f"You are an assistant. Answer the user query clearly and concisely.\n\nUser: {prompt_text}",
                    temperature=0.3,
                    max_output_tokens=200
                )
                return getattr(response, 'text', str(response))
            except Exception as inner_e:
                return f"⚠️ Error calling fallback LLM: {e} | {inner_e}"

    return llm_query