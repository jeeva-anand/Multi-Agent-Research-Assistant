
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma



import streamlit as st
import chromadb
import tempfile
import os


def retrieve_data(uploaded_files):
  doc = []
  temp_dir = tempfile.TemporaryDirectory()
  
  for file in uploaded_files:
    temp_filepath = os.path.join(temp_dir.name, file)
    
    with open(temp_filepath, "wb") as f:
      f.write(file.getValue())
    
    loader = PyMuPDFLoader(temp_filepath)
    doc.extend(loader.load())
    
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
   
    doc_chunks = text_splitter.split_documents(doc)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path="./chroma_db")    

    vectordb = Chroma.from_documents(doc_chunks,embeddings, client=client, collection_name="my_collection")
    
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    
    