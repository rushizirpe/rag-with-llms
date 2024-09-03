# src/langchain/embeddings.py
from sentence_transformers import SentenceTransformer
import openai

class Embeddings:
    def __init__(self, openai_api_key='<OPENAPI_KEY>', embed_model_id = 'all-MiniLM-L6-v2'):
        # Set up OpenAI API key
        openai.api_key = openai_api_key

        # Initialize SentenceTransformer model
        self.embed_model = SentenceTransformer(embed_model_id)

    def get_model():
        return self.embed_model
        
    def filter_citations_and_links(self, text):
        # Remove citations like [1], [2], ...
        text_no_citations = re.sub(r'\[\d+\]', '', text)

        # Remove links
        text_no_links = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
                            '', text_no_citations)

        # Remove www links
        text_no_links = re.sub(r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
                            '', text_no_links)

        return text_no_links

    def create_openai_embeddings(self, sentences):
        # Create OpenAI embeddings for the given sentences
        res = openai.Embedding.create(
            input=sentences,
            engine="text-embedding-ada-002"
        )
        embeddings = [entry['embedding'] for entry in res['data']]
        return embeddings

    def create_sentence_transformer_embeddings(self, sentences):
        # Create SentenceTransformer embeddings for the given sentences
        embeddings = self.embed_model.encode(sentences)
        return embeddings
