import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings 

api_key = st.secrets["GEMINI_API_KEY"]

if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable not set. Please set it in your .env file.")

try:
    llm = ChatGoogleGenerativeAI(model="gemma-3-12b-it", temperature=1, google_api_key=api_key)
except Exception as e:
    print(f"Error initializing the LLM: {e}")
    llm = None

try:
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
except Exception as e:
    print(f"Error initializing the embedding model: {e}")
    embeddings = None

if llm and embeddings:
    print("LLM and Embedding models loaded successfully.")