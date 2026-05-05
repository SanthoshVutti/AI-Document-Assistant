import os
import streamlit as st
import openai

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


# -----------------------
# API KEY
# -----------------------


openai.api_key =  os.getenv("OPENAI_API_KEY")


# -----------------------
# Streamlit UI
# -----------------------

st.set_page_config(
    page_title="AI Document Assistant",
    layout="centered"
)

st.title(
    "AI-Powered Document Assistant"
)

st.write(
"Upload a PDF and ask questions from the document"
)


os.makedirs(
"uploads",
exist_ok=True
)

os.makedirs(
"chroma_db",
exist_ok=True
)


uploaded_file = st.file_uploader(
"Choose PDF File",
type=["pdf"]
)


if uploaded_file:

    filepath = os.path.join(
        "uploads",
        uploaded_file.name
    )

    with open(
        filepath,
        "wb"
    ) as f:

        f.write(
            uploaded_file.getbuffer()
        )


    st.success(
        "Document uploaded successfully"
    )


    with st.spinner(
        "Processing document..."
    ):

        try:

            loader = PyPDFLoader(
                filepath
            )

            docs = loader.load()


            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )

            chunks = splitter.split_documents(
                docs
            )


            embeddings = HuggingFaceEmbeddings(
                 model_name="sentence-transformers/all-mpnet-base-v2"
            )


            vectordb = Chroma.from_documents(
                chunks,
                embedding=embeddings,
                persist_directory="chroma_db"
            )


            st.session_state.retriever = vectordb.as_retriever()

            st.success(
                "Document processed successfully"
            )

        except Exception as e:

            st.error(
                str(e)
            )


    question = st.text_input(
        "Ask a question from document:"
    )


    if st.button("Ask"):

        if not question:

            st.warning(
                "Please enter a question"
            )

        else:

            try:

                docs = retriever.invoke(question)
                

                context = "\n".join(
                    [
                        d.page_content
                        for d in docs
                    ]
                )
                st.subheader("Retrieved Context (Sources)")
                for doc in docs:
                     st.write(doc.page_content[:200])


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


                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {
                            "role":"user",
                            "content":prompt
                        }
                    ]
                )


                answer = response[
                    "choices"
                ][0][
                    "message"
                ][
                    "content"
                ]


                st.subheader(
                    "Answer"
                )

                st.write(
                    answer
                )


            except Exception as e:

                st.error(
                    str(e)
                )
