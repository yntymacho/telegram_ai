from typing import List, Dict, Any
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from loguru import logger
from config import config

class VectorStore:
    def __init__(self):
        """Initialize ChromaDB with embedding function"""
        try:
            self.client = chromadb.PersistentClient(path=str(config.CHROMA_DB_PATH))
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=config.EMBEDDING_MODEL
            )
            self.collection = self.client.get_or_create_collection(
                name="sales_data",
                embedding_function=self.embedding_function
            )
            logger.info("Vector store initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {str(e)}")
            raise

    def load_data(self, df: pd.DataFrame) -> None:
        """Load QA pairs into vector store"""
        try:
            # Clear existing data
            self.collection.delete(where={})
            
            documents = df['question'].tolist()
            metadatas = [{'answer': ans} for ans in df['answer']]
            ids = [f"doc_{i}" for i in range(len(documents))]
            
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            logger.info(f"Loaded {len(documents)} documents into vector store")
        except Exception as e:
            logger.error(f"Failed to load data into vector store: {str(e)}")
            raise

    def search(self, query: str, n_results: int = None) -> List[Dict[str, Any]]:
        """Search for relevant QA pairs"""
        try:
            n_results = n_results or config.TOP_K_RESULTS
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )
            
            # Format results
            answers = []
            for doc, metadata, distance in zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            ):
                answers.append({
                    'question': doc,
                    'answer': metadata['answer'],
                    'relevance_score': 1 - distance  # Convert distance to similarity score
                })
            
            logger.debug(f"Found {len(answers)} relevant answers for query: {query[:50]}...")
            return answers
            
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            return []  # Return empty list on error
