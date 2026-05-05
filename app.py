import os
import shutil
import streamlit as st
import openai

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


# -----------------------
# API KEY
# -----------------------
openai.api_key = os.getenv("OPENAI_API_KEY")


# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="AI Document Assistant", layout="centered")

st.title("AI-Powered Document Assistant")
st.write("Upload a PDF and ask questions from the document")


# -----------------------
# Session State Init
# -----------------------
if "retriever" not in st.session_state:
    st.session_state.retriever = None


# -----------------------
# Create folders
# -----------------------
os.makedirs("uploads", exist_ok=True)


# -----------------------
# Upload PDF
# -----------------------
uploaded_file = st.file_uploader("Choose PDF File", type=["pdf"])


# -----------------------
# Process Document
# -----------------------
if uploaded_file:

    filepath = os.path.join("uploads", uploaded_file.name)

    with open(filepath, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("Document uploaded successfully")

    with st.spinner("Processing document..."):
        try:
            # 🔥 CLEAR OLD VECTOR DATABASE
            if os.path.exists("chroma_db"):
                shutil.rmtree("chroma_db")

            loader = PyPDFLoader(filepath)
            docs = loader.load()

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )

            chunks = splitter.split_documents(docs)

            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-mpnet-base-v2"
            )

            vectordb = Chroma.from_documents(
                chunks,
                embedding=embeddings,
                persist_directory="chroma_db"
            )

            # Store retriever in session
            st.session_state.retriever = vectordb.as_retriever()

            st.success("Document processed successfully")

        except Exception as e:
            st.error(f"Error: {str(e)}")


# -----------------------
# Ask Question
# -----------------------
question = st.text_input("Ask a question from document:")

if st.button("Ask"):

    if not question:
        st.warning("Please enter a question")

    elif st.session_state.retriever is None:
        st.warning("Please upload a document first")

    else:
        try:
            docs = st.session_state.retriever.invoke(question)

            context = "\n".join([doc.page_content for doc in docs])

            # Show retrieved chunks
            st.subheader("Retrieved Context (Sources)")
            for i, doc in enumerate(docs):
                with st.expander(f"Source {i+1}"):
                    st.write(doc.page_content)

            # Prompt
            prompt = f"""
You are an AI assistant answering questions from a document.

Rules:
- Answer only from context
- If answer not found, say "Not found in document"
- Be clear and concise

Context:
{context}

Question:
{question}
"""

            # OpenAI API call
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}]
            )

            answer = response["choices"][0]["message"]["content"]

            # Display answer
            st.subheader("Answer")
            st.write(answer)

        except Exception as e:
            st.error(f"Error: {str(e)}")
