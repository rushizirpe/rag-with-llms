# src/main.py
from src.langchain_components.llms import HuggingFacePipeline
from src.langchain_components.chains import RetrievalQA
from src.langchain_components.vectorstores import annoy, pinecone, faiss
from src.utils.helpers import filter_citations_and_links
from src.langchain_components import embedding_instance, hugging_face_pipeline_instance
import pandas as pd

def load_dataset(file_path):
    df = pd.read_csv(file_path)
    sentences = df['text'].tolist()
    return sentences

def main():
    # Initialize Hugging Face LLM pipeline
    hf_pipeline = hugging_face_pipeline_instance

    # Load and preprocess dataset
    dataset_path = "data/wiki_200k.csv"
    sentences = load_dataset(dataset_path)

    # Filter citations and links from sentences
    filtered_sentences = [filter_citations_and_links(sentence) for sentence in sentences]

    # Initialize Annoy vector store
    annoy_vectorstore = annoy.AnnoyVectorStore(filtered_sentences, embedding_instance)

    # Initialize Pinecone vector store
    # (Note: You need to replace 'YOUR_PINECONE_API_KEY' with your actual Pinecone API key)
    #pinecone_vectorstore = pinecone("llama-2-rag", hf_pipeline.embed_query, "text")

    # Initialize FAISS vector store
    faiss_vectorstore = faiss.FaissVectorStore(filtered_sentences, embedding_instance)

    # Initialize RetrievalQA chain with Annoy vector store
    retrievalqa_annoy = RetrievalQA.from_chain_type(
        llm=hf_pipeline, chain_type='stuff', retriever=annoy_vectorstore.as_retriever()
    )

    # Initialize RetrievalQA chain with Pinecone vector store
    retrievalqa_pinecone = RetrievalQA.from_chain_type(
        llm=hf_pipeline, chain_type='stuff', retriever=pinecone_vectorstore.as_retriever()
    )

    # Initialize RetrievalQA chain with FAISS vector store
    retrievalqa_faiss = RetrievalQA.from_chain_type(
        llm=hf_pipeline, chain_type='stuff', retriever=faiss_vectorstore.as_retriever()
    )

    # Example usage of the RetrievalQA chains
    response_annoy = retrievalqa_annoy("What are Applications of Large Language Models?")
    print(response_annoy['result'])

    response_pinecone = retrievalqa_pinecone("What are Applications of Large Language Models?")
    print(response_pinecone['result'])

    response_faiss = retrievalqa_faiss("What are Applications of Large Language Models?")
    print(response_faiss['result'])

if __name__ == "__main__":
    main()
