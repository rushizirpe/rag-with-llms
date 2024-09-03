# src/langchain/vectorstores/pinecone.py
import os
import time
import pinecone

class PineconeVectorStore:
    def __init__(self, embeddings, index_name='llama-2-rag'):
        self.embeddings = embeddings
        self.index_name = index_name
        self.api_key = os.environ.get('PINECONE_API_KEY') or 'YOUR_PINECONE_API_KEY'
        self.environment = os.environ.get('PINECONE_ENVIRONMENT') or 'YOUR_PINECONE_ENVIRONMENT'
        self.index = None

        pinecone.init(api_key=self.api_key, environment=self.environment)

    def create_index(self):
        if self.index_name not in pinecone.list_indexes():
            pinecone.create_index(
                self.index_name,
                dimension=len(self.embeddings[0]),
                metric='cosine'
            )
            # wait for index to finish initialization
            while not pinecone.describe_index(self.index_name).status['ready']:
                time.sleep(1)

    def connect_to_index(self):
        self.index = pinecone.Index(self.index_name)
        print(self.index.describe_index_stats())

    def embed_and_index_documents(self, data, batch_size=32):
        data = data.to_pandas()

        for i in range(0, len(data), batch_size):
            i_end = min(len(data), i + batch_size)
            batch = data.iloc[i:i_end]
            ids = [f"{x['doi']}-{x['chunk-id']}" for _, x in batch.iterrows()]
            texts = [x['chunk'] for _, x in batch.iterrows()]
            embeds = self.embeddings.embed_documents(texts)
            metadata = [
                {'text': x['chunk'],
                 'source': x['source'],
                 'title': x['title']} for _, x in batch.iterrows()
            ]
            self.index.upsert(vectors=zip(ids, embeds, metadata))

    def similarity_search(self, query, num_samples=5):
        return self.index.similarity_search(query, k=num_samples)
