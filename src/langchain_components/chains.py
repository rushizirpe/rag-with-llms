# src/langchain/chains.py
from langchain.vectorstores import FAISS

class RetrievalQA:
    def __init__(self, llm, retriever=None):
        self.llm = llm
        self.retriever = retriever or FAISS()  # Default to Annoy if no retriever is provided

    def __call__(self, query):
        # Retrieve relevant documents
        relevant_documents = self.retriever.similarity_search(query, k=3)  # Adjust 'k' based on your needs

        # Generate response using the language model
        response = self.llm(query)

        return {
            'query': query,
            'relevant_documents': relevant_documents,
            'result': response
        }

# Usage:
# rag_pipeline = RetrievalQA(llm=my_llm_instance, retriever=my_retriever_instance)
# response = rag_pipeline("What are Applications of Large Language Models?")
# print(response['result'])
