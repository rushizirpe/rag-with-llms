# src/langchain/__init__.py
"""
LangChain Initialization Module

This module can be used for any necessary initialization for the LangChain package.
"""

# Importing modules from langchain package
from .embeddings import Embeddings
from .llms import HuggingFacePipeline
from .chains import RetrievalQA
from .vectorstores import annoy, pinecone, faiss

# Initializing LangChain components
embedding_instance = Embeddings()

hugging_face_pipeline_instance = HuggingFacePipeline("deepset/roberta-base-squad2")
# annoy_vectorstore_instance = annoy.AnnoyVectorStore()
# pinecone_vectorstore_instance = pinecone.PineconeVectorStore()
# faiss_vectorstore_instance = faiss.FaissVectorStore()
# retrieval_qa_instance = RetrievalQA(hugging_face_pipeline_instance, )

# #