
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
    file_path = os.path.join(temp_dir.name,file)
    
    with open(file_path, "wb") as f:
      f.write(file.getValue())
    
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
  embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    

print(os.path)