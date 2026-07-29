

from langchain_community.document_loaders import PyMuPDFLoader

import streamlit as st
import chromadb
import tempfile
import os



def retrieve_data():
  doc = []
  temp_dir = tempfile.TemporaryDirectory()
  
  