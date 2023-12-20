# src/langchain/vectorstores/annoy.py
from annoy import AnnoyIndex
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Annoy

class AnnoyVectorStore:
    def __init__(self, texts, embed_model_hf):
        self.vector_dim = 384  # Dimension of BERT embeddings
        self.num_trees = 10  # Number of trees in the index
        self.vectorstore = Annoy.from_texts(texts, HuggingFaceEmbeddings())
        self.annoy_index = AnnoyIndex(self.vector_dim, 'angular')  # 'angular' - similarity metric
        self.embed_model_hf = embed_model_hf
        self.build_annoy_index(texts)

    def get_store():
        return self.vectorstore
    
    def build_annoy_index(self, texts):
        embeddings = self.embed_model_hf.embed_documents(texts)
        for doc_id, doc_embedding in enumerate(embeddings):
            self.annoy_index.add_item(doc_id, doc_embedding)
        self.annoy_index.build(self.num_trees)

    def similarity_search(self, query, num_samples=5):
        query_embedding = self.embed_model_hf.embed_documents([query])[0]
        similar_doc_ids = self.annoy_index.get_nns_by_vector(query_embedding, num_samples)

        # Return the indices of similar documents
        return similar_doc_ids

# Usage:
# Instantiate AnnoyVectorStore
# texts = ["Text 1", "Text 2", "Text 3"]  # Replace with your actual text data
# embed_model_hf = HuggingFaceEmbeddings(model_name='your_embedding_model_name')
# annoy_vectorstore = AnnoyVectorStore(texts, embed_model_hf)

# Perform similarity search
# query_text = "Your query text here"
# similar_documents = annoy_vectorstore.similarity_search(query_text)
# print("Similar Documents:", similar_documents)
