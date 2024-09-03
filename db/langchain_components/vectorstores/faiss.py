# src/langchain/vectorstores/faiss.py
import faiss
from langchain.vectorstores import FAISS
import numpy as np

class FaissVectorStore:
    def __init__(self, texts, embed_model_hf, embeddings = "<SKIP>" ):
        self.embeddings = embeddings
        #self.vector_dim = len(embeddings[0])
        self.index = self.build_index()
        self.vectorstore = FAISS.from_texts(texts, embed_model_hf)

    def build_index(self):
        index = faiss.IndexFlatL2(self.vector_dim)
        embeddings_array = np.array(self.embeddings, dtype=np.float32)
        index.add(embeddings_array)
        return index
    
    def get_store():
        return self.vectorstore

    def similarity_search(self, query, num_samples=3):
        query_embedding = self.embed_model_hf.encode([query])[0]
        D, I = self.index.search(np.array([query_embedding]), k=num_samples)
        return I[0]

# Example usage:
# Instantiate FaissVectorStore with your embeddings
# embeddings = ...  # Replace with your actual embeddings
# faiss_store = FaissVectorStore(embeddings)

# Perform similarity search
# query = "Your search query"
# result = faiss_store.similarity_search(query)
# print(result)
