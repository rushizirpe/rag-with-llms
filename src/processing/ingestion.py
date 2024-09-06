# src/data_ingestion/ingestion.py

import logging
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def ingest_data(data_dir, persist_dir):
    try:
        documents = SimpleDirectoryReader(data_dir).load_data()
        storage_context = StorageContext.from_defaults()
        index = VectorStoreIndex.from_documents(documents)
        index.storage_context.persist(persist_dir=persist_dir)
        logger.info(f"Data ingested successfully from {data_dir}")
    except Exception as e:
        logger.error(f"Error during data ingestion: {str(e)}")
        raise