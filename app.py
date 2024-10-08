import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from chromadb.config import Settings
import chromadb

# Loading Environment Variables
load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

# Problem: An error occurred: Could not connect to tenant default_tenant. Are you sure it exists?
# Solution: 
chromadb.api.client.SharedSystemClient.clear_system_cache()

# Embedding Model
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Streamlit App
st.title("Conversational RAG with PDF Uploads and Chat History")
st.write("Upload PDFs and chat with their content")

groq_api_key = st.text_input("Enter your Groq API Key:", type="password")  # Groq API Key Input

if groq_api_key:
    # Initializing the Language Model (LLM)
    llm = ChatGroq(model_name="gemma2-9b-it", groq_api_key=groq_api_key)

    session_id = st.text_input("Session ID:", value="default_session")  

    if 'store' not in st.session_state:
        st.session_state.store = {}

    uploaded_files = st.file_uploader("Choose a PDF file", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        try:
            # Ensure 'temp' directory exists
            if not os.path.exists('temp'):
                os.makedirs('temp')

            # a. Load PDF Files
            documents = []

            for uploaded_file in uploaded_files:
                pdf_path = os.path.join("temp", uploaded_file.name)  # Use uploaded_file.name for uniqueness

                with open(pdf_path, "wb") as file:
                    file.write(uploaded_file.getvalue())

                pdf_documents = PyPDFLoader(pdf_path).load()  # Load Documents
                documents.extend(pdf_documents)

            # b. Split the document
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            final_text_documents = text_splitter.split_documents(documents)

            # c. Storing in Chroma DB (Vector Store)
            vectorStore = Chroma.from_documents(
                documents=final_text_documents, 
                embedding=embedding,
            )
            retriever = vectorStore.as_retriever()

            # Adding Chat History
            contextualize_q_system_prompt = (
                "Given a chat history and the latest user question "
                "which might reference context in the chat history, "
                "formulate a standalone question which can be understood "
                "without the chat history. Do NOT answer the question, "
                "just reformulate it if needed and otherwise return it as is."
            )

            contextualize_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", contextualize_q_system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}")
                ]
            )

            # History Retriever
            history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_prompt)

            # Answer context prompt
            system_prompt = (
                "You are an assistant for question-answering tasks."
                "Use the following pieces of retrieved context to answer "
                "the question. If you don't know the answer, say that you "
                "don't know. Use three sentences maximum and keep the "
                "answer concise."
                "\n\n"
                "{context}"
            )

            question_answer_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}")
                ]
            )

            # Summarize Chain
            question_answer_chain = create_stuff_documents_chain(llm, question_answer_prompt)

            # Research Assistant
            rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

            def get_session_history(session_id: str) -> BaseChatMessageHistory:
                if session_id not in st.session_state.store:
                    st.session_state.store[session_id] = ChatMessageHistory()
                return st.session_state.store[session_id]

            with_message_history = RunnableWithMessageHistory(
                rag_chain,
                get_session_history,
                input_messages_key="input",
                history_messages_key="chat_history",
                output_messages_key="answer"
            )

            # Take user input
            user_input = st.text_input("Your Question:")

            if user_input:
                session_history = get_session_history(session_id)

                response = with_message_history.invoke(
                    {"input": user_input},
                    config={"configurable": {"session_id": session_id}}
                )

                st.write("Assistant:", response['answer'])

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

else:
    st.warning("Please enter your Groq API Key to continue")
