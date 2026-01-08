# src/retriever.py
"""
Simple vector retriever for your collected data
"""

import json
from pathlib import Path
from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorRetriever:
    def __init__(self):
        self.data = self._load_data()
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Pre-compute embeddings if not already done
        self.embeddings = self._load_or_create_embeddings()
    
    def _load_data(self) -> List[Dict]:
        """Load cleaned data"""
        try:
            chunks_path = Path("data/cleaned/chunks.json")
            if chunks_path.exists():
                with open(chunks_path, 'r') as f:
                    return json.load(f)
            
            # Fallback to collected raw data
            raw_dir = Path("data/collected/raw")
            all_data = []
            for file_path in raw_dir.glob("*.json"):
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            all_data.extend(data)
                        elif isinstance(data, dict):
                            all_data.append(data)
                except:
                    continue
            
            return all_data
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return []
    
    def _load_or_create_embeddings(self):
        """Load or create embeddings for the data"""
        embeddings_path = Path("data/embeddings/embeddings.npy")
        
        if embeddings_path.exists():
            # Load existing embeddings
            return np.load(embeddings_path, allow_pickle=True).item()
        else:
            # Create new embeddings
            texts = []
            for item in self.data:
                if isinstance(item, dict):
                    text = item.get('text') or item.get('content') or item.get('abstract') or ''
                    if text:
                        texts.append(str(text)[:1000])  # Limit text length
            
            if texts:
                logger.info(f"Creating embeddings for {len(texts)} texts")
                embeddings = self.model.encode(texts, show_progress_bar=True)
                
                # Save for future use
                embeddings_path.parent.mkdir(exist_ok=True)
                np.save(embeddings_path, {'embeddings': embeddings, 'texts': texts})
                
                return {'embeddings': embeddings, 'texts': texts}
            else:
                return {'embeddings': np.array([]), 'texts': []}
    
    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """Search for relevant information"""
        if not self.data or len(self.embeddings['embeddings']) == 0:
            return []
        
        # Encode query
        query_embedding = self.model.encode([query])[0]
        
        # Calculate similarities
        similarities = np.dot(self.embeddings['embeddings'], query_embedding)
        
        # Get top results
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if idx < len(self.data):
                item = self.data[idx]
                if isinstance(item, dict):
                    text = item.get('text') or item.get('content') or item.get('abstract') or ''
                    if text:
                        results.append({
                            'text': str(text)[:500],  # Limit response length
                            'source': item.get('source_type', 'medical source'),
                            'score': float(similarities[idx]),
                            'metadata': {k: v for k, v in item.items() if k not in ['text', 'content', 'abstract']}
                        })
        
        return results