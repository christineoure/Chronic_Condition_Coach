# embeddings_vector_db.py
"""
Create embeddings from cleaned text chunks and store in vector database
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
from typing import List, Dict, Any
import sys

# Import vector database library
try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    print("ChromaDB not installed. Install with: pip install chromadb")

# Import embedding model
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    print("SentenceTransformers not installed. Install with: pip install sentence-transformers")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorDBManager:
    """Manage vector database operations"""
    
    def __init__(self, persist_directory: str = "vector_db"):
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(exist_ok=True)
        
        if not CHROMA_AVAILABLE:
            raise ImportError("ChromaDB is not installed")
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Load embedding model
        if EMBEDDINGS_AVAILABLE:
            logger.info("Loading embedding model...")
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            self.embedding_dim = 384
        else:
            self.model = None
            self.embedding_dim = 768  # Default for OpenAI embeddings
            logger.warning("Using mock embeddings. Install sentence-transformers for real embeddings.")
    
    def create_collection(self, collection_name: str = "chronic_conditions"):
        """Create or get a collection"""
        try:
            # Try to get existing collection
            collection = self.client.get_collection(collection_name)
            logger.info(f"Using existing collection: {collection_name}")
        except:
            # Create new collection
            collection = self.client.create_collection(
                name=collection_name,
                metadata={"description": "Chronic condition medical documents"}
            )
            logger.info(f"Created new collection: {collection_name}")
        
        self.collection = collection
        return collection
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for texts"""
        if self.model:
            # Use SentenceTransformers
            embeddings = self.model.encode(texts, show_progress_bar=True)
            return embeddings.tolist()
        else:
            # Mock embeddings for testing
            logger.warning("Generating mock embeddings - install sentence-transformers for real embeddings")
            return [list(np.random.randn(self.embedding_dim)) for _ in texts]
    
    def add_documents_to_collection(self, chunks_df: pd.DataFrame, batch_size: int = 100):
        """Add documents to vector database in batches"""
        if self.collection is None:
            self.create_collection()
        
        total_chunks = len(chunks_df)
        logger.info(f"Adding {total_chunks} chunks to vector database...")
        
        # Process in batches
        for start_idx in range(0, total_chunks, batch_size):
            end_idx = min(start_idx + batch_size, total_chunks)
            batch_df = chunks_df.iloc[start_idx:end_idx]
            
            # Prepare batch data
            ids = batch_df['chunk_id'].tolist()
            texts = batch_df['text'].tolist()
            
            # Generate embeddings
            embeddings = self.generate_embeddings(texts)
            
            # Prepare metadata
            metadatas = []
            for _, row in batch_df.iterrows():
                metadata = {
                    'source_type': row.get('source_type', 'unknown'),
                    'topic': row.get('topic', 'general'),
                    'document_index': str(row.get('document_index', 0)),
                    'chunk_index': str(row.get('chunk_index', 0)),
                    'word_count': str(row.get('word_count', 0)),
                    'title': row.get('title', '')[:200] if pd.notna(row.get('title')) else ''
                }
                metadatas.append(metadata)
            
            # Add to collection
            try:
                self.collection.add(
                    embeddings=embeddings,
                    documents=texts,
                    metadatas=metadatas,
                    ids=ids
                )
                logger.info(f"Added batch {start_idx//batch_size + 1}: {len(ids)} documents")
                
            except Exception as e:
                logger.error(f"Error adding batch {start_idx//batch_size + 1}: {e}")
                # Try without embeddings (let ChromaDB generate them)
                try:
                    self.collection.add(
                        documents=texts,
                        metadatas=metadatas,
                        ids=ids
                    )
                    logger.info(f"Added batch {start_idx//batch_size + 1} without precomputed embeddings")
                except Exception as e2:
                    logger.error(f"Failed to add batch even without embeddings: {e2}")
        
        logger.info(f"Successfully added {total_chunks} documents to vector database")
    
    def query_collection(self, query_text: str, n_results: int = 5) -> List[Dict]:
        """Query the vector database"""
        if self.collection is None:
            self.create_collection()
        
        # Generate embedding for query
        if self.model:
            query_embedding = self.model.encode([query_text])[0].tolist()
        else:
            query_embedding = list(np.random.randn(self.embedding_dim))
        
        # Query the collection
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        # Format results
        formatted_results = []
        if results['documents']:
            for i in range(len(results['documents'][0])):
                result = {
                    'text': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                    'distance': results['distances'][0][i] if results['distances'] else 0,
                    'id': results['ids'][0][i] if results['ids'] else f"result_{i}"
                }
                formatted_results.append(result)
        
        return formatted_results
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about the collection"""
        if self.collection is None:
            return {"error": "No collection loaded"}
        
        try:
            # Count documents (this might be approximate for large collections)
            count = self.collection.count()
            
            # Get sample metadata
            sample = self.collection.peek(limit=10)
            
            stats = {
                'total_documents': count,
                'embedding_dimension': self.embedding_dim,
                'collection_name': self.collection.name,
                'sample_size': len(sample['ids']) if sample['ids'] else 0
            }
            
            return stats
        except Exception as e:
            return {"error": str(e)}


def main():
    """Main function to create embeddings and populate vector database"""
    print(" Starting embeddings and vector database creation...")
    
    # Check dependencies
    if not CHROMA_AVAILABLE:
        print(" ChromaDB not installed. Run: pip install chromadb")
        return
    
    # Load cleaned data
    cleaned_dir = Path("data/cleaned")
    chunks_json = cleaned_dir / "chunks.json"
    chunks_csv = cleaned_dir / "chunks.csv"
    
    if not chunks_json.exists() and not chunks_csv.exists():
        print(" No cleaned data found. Run run_cleaning_chunking.py first.")
        return
    
    try:
        # Load data
        if chunks_json.exists():
            chunks_df = pd.read_json(chunks_json)
        else:
            chunks_df = pd.read_csv(chunks_csv)
        
        print(f" Loaded {len(chunks_df)} text chunks")
        
        # Initialize vector database
        db_manager = VectorDBManager()
        
        # Create collection
        db_manager.create_collection("chronic_conditions_v2")
        
        # Add documents
        db_manager.add_documents_to_collection(chunks_df, batch_size=50)
        
        # Get stats
        stats = db_manager.get_collection_stats()
        print("\n" + "="*50)
        print(" VECTOR DATABASE CREATED SUCCESSFULLY")
        print("="*50)
        print(f"Total documents in database: {stats.get('total_documents', 'N/A')}")
        print(f"Embedding dimension: {stats.get('embedding_dimension', 'N/A')}")
        print(f"Collection name: {stats.get('collection_name', 'N/A')}")
        
        # Test query
        print("\n Testing with sample query...")
        test_queries = [
            "What foods are good for diabetes?",
            "How to lower blood pressure naturally?",
            "Managing chronic pain"
        ]
        
        for query in test_queries[:1]:  # Test first query only
            results = db_manager.query_collection(query, n_results=3)
            print(f"\nQuery: '{query}'")
            print(f"Found {len(results)} results")
            if results:
                print(f"Top result (first 200 chars): {results[0]['text'][:200]}...")
        
        print("\n Next step: Start your Streamlit app: streamlit run app_simple.py")
        
    except Exception as e:
        logger.error(f"Error creating vector database: {e}")
        raise


if __name__ == "__main__":
    main()