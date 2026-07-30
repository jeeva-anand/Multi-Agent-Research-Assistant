import streamlit as st
import chromadb
import tempfile
import os
import pandas as pd

from operator import itemgetter

from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.callbacks.base import BaseCallbackHandler

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_chroma import Chroma
from langchain_community.chat_message_histories import StreamlitChatMessageHistory



st.set_page_config(
    page_title="ResearchPilot AI",
    page_icon="",
    layout="wide"
)


st.markdown(
    """
<style>

.main {
    background-color:#0E1117;
}


h1 {
    font-size:45px !important;
    font-weight:800;
}


.subtitle {

    color:#9CA3AF;
    font-size:18px;

}


.card {

    background:#161B22;
    padding:20px;
    border-radius:15px;
    border:1px solid #30363D;

}


.status {

    padding:10px;
    border-radius:10px;
    background:#111827;

}


</style>

""",
    unsafe_allow_html=True
)




st.markdown(
    """
#  ResearchPilot AI

<div class="subtitle">
Multi-Agent RAG Research Assistant powered by Llama 3.2, ChromaDB and LangChain
</div>

<br>

""",
    unsafe_allow_html=True
)



message_history = StreamlitChatMessageHistory(
    key="research_chat"
)



@st.cache_resource(ttl="1h")
def create_vector_db(files):

    documents = []

    temp_dir = tempfile.TemporaryDirectory()

    for file in files:

        path = os.path.join(
            temp_dir.name,
            file.name
        )

        with open(path, "wb") as f:
            f.write(file.getvalue())

        loader = PyMuPDFLoader(path)

        documents.extend(
            loader.load()
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(
        documents
    )

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    client = chromadb.PersistentClient(
        path="./chroma_db"
    )

    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        client=client,
        collection_name="researchpilot"
    )

    return vector_db.as_retriever(
        search_kwargs={
            "k": 4
        }
    ), len(chunks)




with st.sidebar:

    st.markdown(
        """
    ##  Configuration
    """
    )

    uploaded_files = st.file_uploader(
        "Upload Research Papers",
        type="pdf",
        accept_multiple_files=True
    )

    st.divider()

    st.markdown(
        """
    ###  AI Stack

     Llama 3.2 3B  
     Chroma Vector DB  
     MiniLM Embeddings  
     LangChain RAG  

    """
    )

    st.divider()

    # st.info(
    #     """
    #     Upload PDFs and ask questions.
    #     The assistant retrieves relevant
    #     sections before answering.
    #     """
    # )


if not uploaded_files:

    st.warning(
        "Upload PDF documents to activate ResearchPilot AI"
    )

    st.stop()



retriever, chunk_count = create_vector_db(
    uploaded_files
)


with st.sidebar:

    st.success(
        f"📚 {chunk_count} chunks indexed"
    )



llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0,
    streaming=True
)



prompt = ChatPromptTemplate.from_template(

    """
You are an expert research assistant.

Use only the provided context.

Conversation history:
{history}


Context:
{context}


Question:
{question}


Provide a clear technical answer.
"""
)


def format_docs(docs):

    return "\n\n".join(
        doc.page_content
        for doc in docs
    )


def format_history():

    text = ""

    for msg in message_history.messages:

        text += (
            f"{msg.type}: "
            f"{msg.content}\n"
        )

    return text




for msg in message_history.messages:

    st.chat_message(
        msg.type
    ).write(
        msg.content
    )



question = st.chat_input(
    "Ask about your research papers..."
)


if question:

    st.chat_message(
        "user"
    ).write(question)

    message_history.add_user_message(
        question
    )

    with st.chat_message(
        "assistant"
    ):

        with st.spinner(
            "🔎 Searching documents..."
        ):

            docs = retriever.invoke(
                question
            )

            context = format_docs(
                docs
            )

            response = llm.invoke(

                prompt.format(

                    history=format_history(),

                    context=context,

                    question=question
                )

            )

        st.write(
            response.content
        )

        with st.expander(
            "📚 Sources"
        ):

            source_data = []

            for d in docs:

                source_data.append(
                    {
                        "File":
                        d.metadata.get("source"),

                        "Page":
                        d.metadata.get("page"),

                        "Preview":
                        d.page_content[:200]
                    }
                )

            st.dataframe(
                pd.DataFrame(source_data)
            )

    message_history.add_ai_message(
        response.content
    )
