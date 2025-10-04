# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.chains import ConversationalRetrievalChain
# from langchain.memory import ConversationBufferMemory
# from src.config import CHAT_MODEL

# def create_chatbot(vectorstore):
#     retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
#     llm = ChatGoogleGenerativeAI(model=CHAT_MODEL, temperature=0.2)
    
#     memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
#     qa_chain = ConversationalRetrievalChain.from_llm(
#         llm=llm,
#         retriever=retriever,
#         memory=memory
#     )
#     return qa_chain




from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from src.config import CHAT_MODEL, GEMINI_API_KEY


def create_chatbot(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    llm = ChatGoogleGenerativeAI(
        model=CHAT_MODEL,
        temperature=0.2,
        google_api_key=GEMINI_API_KEY   # ✅ API key force
    )
    
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory
    )
    return qa_chain
