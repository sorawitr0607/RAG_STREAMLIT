import pysqlite3
import sys

sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st

# ---- RAG & LangChain imports ----
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings


# Streamlit config & page layout
st.set_page_config(layout='centered', page_title="TGH Agent Chatbot", page_icon="🤖")
st.title('🤖 Agent Chatbot')

#--------------------------------------
# Sidebar
#--------------------------------------
st.sidebar.image('logo.png')
st.markdown(
    """
    <style>
        div[data-testid="stSidebarUserContent"] img {
            background: #ffffff;
            border-radius: 20px;
            border: thick;
            border-style: inset;
            border-color: black;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.sidebar.header("About")
st.sidebar.markdown(
    """
    🤖 A Large Language Model optimized for Thai.
    """
)
st.sidebar.markdown(
    "[Streamlit](https://streamlit.io) is a Python library that allows the creation of interactive, data-driven web applications."
)
st.sidebar.divider()
st.sidebar.header("Resources")
st.sidebar.markdown(
    """
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Cheat sheet](https://docs.streamlit.io/library/cheatsheet)
- [Book](https://www.amazon.com/dp/180056550X)
- [Blog](https://blog.streamlit.io/how-to-master-streamlit-for-data-science/)
"""
)
#--------------------------------------

@st.cache_resource

def init_rag_components():
    """
    Initialize your embeddings, LLM, Chroma vector store, 
    and prompt template once, and return them.
    """
    # 2.1: Load your custom embeddings (from 'models_test.Models')
    #models = Models()
    #embeddings = OllamaEmbeddings(model="jeffh/intfloat-multilingual-e5-large-instruct:f16")
    embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large"
    ,openai_api_key=st.secrets['OPENAI_API_KEY']
    )

    # 2.2: Setup your LLM from OpenAI (or whichever you use)
    # Be sure to provide your correct OpenAI-compatible key:
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        openai_api_key=st.secrets['OPENAI_API_KEY']
        ,temperature=0.7
    )

    # 2.3: Build your Chroma-based vector store
    vector_store = Chroma(
        collection_name="documents",
        embedding_function=embeddings,
        persist_directory="./chroma_db",
    )

    # 2.4: Build your prompt template for combining docs
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system", 
                "You are a helpful expert insurance salesman agent assistant from 'thai group holdings public company limited'."
                "Your goal is to recommend insurance products based on the given context, "
                "ensuring they match the user's conditions and benefits."
                "Answer the question based only on the data provided."
            ),
            (
                "human",
                "Previous conversation:\n{history}\n"
                "If user question contains relevant insurance information, then use the user question {input} to analyze insurance products."
                "Then, check if the product conditions and benefits match the user's specified conditions and benefits."
                "If they do, summarize the product in a clear and persuasive way, emphasizing its unique advantages."
                "Use only the information in the provided {context} to generate the response. "
                "If user question {input} is unrelated to insurance, respond naturally and briefly. "
                "Respond in Thai."
            ),
        ]
    )

    return llm, vector_store, prompt

#--------------------------------------------------------------------------------

def retrieval_logic(query, llm, vector_store, prompt, chat_history_rag=None):
    """
    Given a user query, retrieve docs from Chroma, then use the chain
    to produce a final answer. 
    """
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    # retrieved_docs = retriever.invoke(query)

    # # # If no docs come back, respond that we lack data
    # if not retrieved_docs:
    #     response = "ขออภัย ฉันไม่มีข้อมูลเกี่ยวกับคำถามนี้ในฐานข้อมูลของฉัน"

    # else:
    # Build the chain on the fly
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
    
    if chat_history_rag is None:
        chat_history_rag = []
    combined_input = {
        "history": "\n".join(chat_history_rag),
        "input": query
    }
            # Run the chain
    result = retrieval_chain.invoke(combined_input)
    response = result["answer"]

    return response

#--------------------------------------------------------------------------------

# 4. Streamlit Chat Loop
#--------------------------------------------------------------------------------
def main():
    # Initialize states
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_history_rag" not in st.session_state:
        st.session_state.chat_history_rag = []
    if "rag_components" not in st.session_state:
        # Cache the model, vector store, and prompt
        st.session_state.rag_components = init_rag_components()
    if "reset" not in st.session_state:
        st.session_state.reset = False

    # 4.1: Display existing messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 4.2: Input for user query
    user_query = st.chat_input("พิมพ์คำถามของคุณที่นี่ ...")

    # 4.3: On user submit
    if user_query:
        # Store user message
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Retrieve RAG components
        llm, vector_store, prompt = st.session_state.rag_components

        # 4.4: Perform RAG retrieval
        rag_answer = retrieval_logic(
            query=user_query,
            llm=llm,
            vector_store=vector_store,
            prompt=prompt,
            chat_history_rag=st.session_state.chat_history_rag
        )
        

        # Update chat_history and store the RAG answer
        st.session_state.chat_history_rag.append(f"User: {user_query}")
        st.session_state.chat_history_rag.append(f"Assistant: {rag_answer}")

        # 4.5: Display the assistant answer
        st.session_state.messages.append({"role": "assistant", "content": rag_answer})
        with st.chat_message("assistant"):
            st.markdown(rag_answer)

        st.session_state.reset = True

    # 4.6: Reset chat button
    if st.session_state.reset:
        st.button("Reset chat", on_click=reset_chat)
        
#--------------------------------------------------------------------------------

def reset_chat():
    """Clear session state to reset the chat."""
    st.session_state.messages = []
    st.session_state.chat_history_rag = []
    st.session_state.reset = False
    

#--------------------------------------------------------------------------------

if __name__ == "__main__":
    main()
