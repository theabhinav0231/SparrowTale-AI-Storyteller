import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from llm_setup import llm, embeddings

if not llm or not embeddings:
    raise ImportError("LLM or Embedding models could not be loaded. Please check llm_setup.py.")

def get_document_context(file_path: str, query: str) -> str:
    """
    Loads a document, splits it, creates an in-memory FAISS vector store,
    and retrieves the most relevant context for a given query.
    """
    print("--- Using FAISS for document retrieval ---")
    
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path)
    else:
        return "Error: Unsupported file format. Please upload a .pdf or .txt file."

    try:
        documents = loader.load()
        if not documents:
            return "Error: Document is empty or could not be read."
            
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        docs = text_splitter.split_documents(documents)
        
        if not docs:
            return "Error: Could not extract text chunks from the document."

        db = FAISS.from_documents(docs, embeddings)
        retriever = db.as_retriever(search_kwargs={'k': 3})
        retrieved_docs = retriever.invoke(query)
        
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        return context
    except Exception as e:
        print(f"An error occurred during document processing: {e}")
        return "Error: Failed to process the provided document."


def run_rag_agent(user_prompt: str, file_path: str) -> str:
    """
    The main agentic function to retrieve context from a user-provided document.
    """
    print("--- RAG Agent Activated (Document-Only Mode) ---")
    
    prompt_template = PromptTemplate.from_template(
        """You are a research assistant. Based on the user's story idea, what is the single most 
        important keyword or question to search for within their provided document to find relevant context?
        
        User's Story Idea: '{prompt}'
        
        Optimized Search Query for Document:"""
    )
    
    chain = prompt_template | llm | StrOutputParser()
    
    search_query = chain.invoke({"prompt": user_prompt})
    print(f"Generated Search Query: {search_query}")

    context = get_document_context(file_path, search_query)
        
    print("--- RAG Agent Finished ---")
    return context