
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
# from langchain.prompts import PromptTemplate
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma



import streamlit as st
import chromadb
import tempfile
import os


st.set_page_config(page_title="Multi-Agent-Research-Assistant", page_icon="")
st.title("Multi-Agent-Research-Assistant")


@st.cache_resource(ttl="1hr")
def retrieve_data(uploaded_files):
  doc = []
  temp_dir = tempfile.TemporaryDirectory()
  
  for file in uploaded_files:
    temp_filepath = os.path.join(temp_dir.name, file.name)    
    with open(temp_filepath, "wb") as f:
      f.write(file.getvalue())    
    loader = PyMuPDFLoader(temp_filepath)
    doc.extend(loader.load())      
      
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)   
    doc_chunks = text_splitter.split_documents(doc)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path="./chroma_db")    
    vectordb = Chroma.from_documents(doc_chunks,embeddings, client=client, collection_name="research_papers")    
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})    
    return retriever
  
  
def StreamHandler(BaseCallbackHandler):
  def __init__(self, container, initial_text=""):
    self.container = container
    self.text = initial_text
    
  def on_llm_new_token(self, token, **kwargs):
    self.text += token
    self.container.markdown(self.text + "")
    


  
uploaded_files = st.sidebar.file_uploader(
    label="Upload PDF files", type=["pdf"],
    accept_multiple_files=True
)
if not uploaded_files:
  st.info("Please upload PDF documents to continue.")
  st.stop()

# Create retriever object based on uploaded PDFs
retriever = retrieve_data(uploaded_files)


llm = ChatOllama(
    model="qwen2.5:7b",
    temperature=0.2)

qa_template = """
              Use only the following pieces of context to answer the question at the end.
              If you don't know the answer, just say that you don't know,
              don't try to make up an answer. Keep the answer as concise as possible.

              {context}

              Question: {question}
              """
              
prompt = ChatPromptTemplate.from_template(qa_template)

def format_docs(docs):
  return "\n\n".join([d.page_content for d in docs])

