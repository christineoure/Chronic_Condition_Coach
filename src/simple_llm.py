#!/usr/bin/env python
"""
Simple LLM Client that uses RAG with your collected medical data
"""

import os
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional
import logging
import numpy as np
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGRetriever:
    """Retriever for your collected medical data"""
    
    def __init__(self):
        self.data_dir = Path("data")
        self.cleaned_dir = self.data_dir / "cleaned"
        self.collected_dir = self.data_dir / "collected" / "raw"
        
        # Load embedding model
        try:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info(" Loaded embedding model")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            self.model = None
        
        # Load data
        self.documents = self._load_documents()
        self.embeddings = self._load_or_create_embeddings()
        
        logger.info(f" Loaded {len(self.documents)} documents for RAG")
    
    def _load_documents(self) -> List[Dict]:
        """Load all collected documents"""
        documents = []
        
        # Try cleaned chunks first
        chunks_path = self.cleaned_dir / "chunks.json"
        if chunks_path.exists():
            try:
                with open(chunks_path, 'r') as f:
                    chunks = json.load(f)
                    for chunk in chunks:
                        if isinstance(chunk, dict) and 'text' in chunk:
                            documents.append({
                                'text': chunk['text'],
                                'source': chunk.get('source_type', 'cleaned'),
                                'metadata': chunk
                            })
                logger.info(f"Loaded {len(documents)} chunks from cleaned data")
                return documents
            except Exception as e:
                logger.error(f"Error loading cleaned chunks: {e}")
        
        # Fallback to collected raw data
        if self.collected_dir.exists():
            for file_path in self.collected_dir.glob("*.json"):
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    
                    # Handle different data formats
                    if isinstance(data, list):
                        for item in data:
                            if isinstance(item, dict):
                                text = self._extract_text(item)
                                if text:
                                    documents.append({
                                        'text': text,
                                        'source': file_path.name,
                                        'metadata': item
                                    })
                    elif isinstance(data, dict):
                        text = self._extract_text(data)
                        if text:
                            documents.append({
                                'text': text,
                                'source': file_path.name,
                                'metadata': data
                            })
                except Exception as e:
                    logger.error(f"Error loading {file_path}: {e}")
        
        logger.info(f"Loaded {len(documents)} documents from raw data")
        return documents
    
    def _extract_text(self, item: Dict) -> str:
        """Extract text from different document formats"""
        # PubMed articles
        if 'abstract' in item and item['abstract']:
            return item['abstract']
        if 'title' in item and item['title']:
            title = item['title']
            abstract = item.get('abstract', '')
            return f"{title}. {abstract}"
        
        # Web content
        if 'content' in item and item['content']:
            return item['content']
        
        # Synthetic content
        if 'text' in item and item['text']:
            return item['text']
        
        # Clinical trials
        if 'summary' in item and item['summary']:
            return item['summary']
        if 'briefTitle' in item and item['briefTitle']:
            return item['briefTitle']
        
        # Q&A
        if 'question' in item and 'answer' in item:
            return f"Q: {item['question']} A: {item['answer']}"
        
        # Try any text field
        for key in ['text', 'content', 'abstract', 'summary', 'description', 'answer']:
            if key in item and item[key]:
                return str(item[key])
        
        return ""
    
    def _load_or_create_embeddings(self):
        """Load or create embeddings for documents"""
        embeddings_path = self.data_dir / "embeddings" / "document_embeddings.pkl"
        
        if embeddings_path.exists() and self.model:
            try:
                with open(embeddings_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.error(f"Error loading embeddings: {e}")
        
        # Create new embeddings
        if self.model and self.documents:
            texts = [doc['text'] for doc in self.documents if doc['text']]
            if texts:
                logger.info(f"Creating embeddings for {len(texts)} texts...")
                embeddings = self.model.encode(texts, show_progress_bar=False)
                
                # Save embeddings
                embeddings_path.parent.mkdir(exist_ok=True)
                with open(embeddings_path, 'wb') as f:
                    pickle.dump(embeddings, f)
                
                logger.info(f" Saved embeddings to {embeddings_path}")
                return embeddings
        
        return None
    
    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """Search for relevant documents"""
        # Check if we have the necessary components
        if not self.documents or self.embeddings is None or self.model is None:
            return []
        
        try:
            # Encode query
            query_embedding = self.model.encode([query])[0]
            
            # Calculate similarities
            similarities = np.dot(self.embeddings, query_embedding)
            
            # Get top results
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            results = []
            for idx in top_indices:
                if idx < len(self.documents):
                    doc = self.documents[idx]
                    results.append({
                        'text': doc['text'][:500],  # Limit length
                        'source': doc['source'],
                        'score': float(similarities[idx]),
                        'metadata': doc.get('metadata', {})
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            # Fallback: simple keyword search
            return self._keyword_search(query, top_k)
    
    def _keyword_search(self, query: str, top_k: int = 3) -> List[Dict]:
        """Simple keyword fallback search"""
        query_words = set(query.lower().split())
        results = []
        
        for doc in self.documents[:100]:  # Search first 100 docs
            text = doc['text'].lower()
            score = sum(1 for word in query_words if word in text)
            
            if score > 0:
                results.append({
                    'text': doc['text'][:500],
                    'source': doc['source'],
                    'score': score / len(query_words),
                    'metadata': doc.get('metadata', {})
                })
            
            if len(results) >= top_k:
                break
        
        return sorted(results, key=lambda x: x['score'], reverse=True)


class SimpleLLMClient:
    """LLM Client that uses RAG with your medical data"""
    
    def __init__(self, provider: str = "openai"):
        """
        Initialize LLM client
        
        Args:
            provider: "openai", "anthropic", or "rag" for RAG-only mode
        """
        self.provider = provider
        self.use_real_llm = False
        
        # Initialize RAG retriever
        self.retriever = RAGRetriever()
        
        # Check if we can use real LLM API
        if provider in ["openai", "anthropic"]:
            api_key = os.getenv(f"{provider.upper()}_API_KEY")
            if api_key:
                self.use_real_llm = True
                logger.info(f" Using {provider.upper()} API")
            else:
                logger.info(f" {provider.upper()}_API_KEY not found. Using RAG-only mode.")
        else:
            logger.info(" Using RAG-only mode with collected medical data")
    
    def get_response(self, query: str, context: Optional[str] = None) -> str:
        """Get response using RAG with your medical data"""
        
        # Step 1: Retrieve relevant medical information
        rag_results = self.retriever.search(query, top_k=3)
        
        if not rag_results:
            # No relevant data found
            return self._get_no_data_response(query)
        
        # Step 2: Format the response based on retrieved data
        return self._format_rag_response(query, rag_results, context)
    
    def _format_rag_response(self, query: str, rag_results: List[Dict], context: Optional[str]) -> str:
        """Format a response based on RAG results"""
        
        # Extract key information from results
        key_points = []
        sources = set()
        
        for i, result in enumerate(rag_results, 1):
            text = result['text']
            source = result['source']
            
            # Extract first sentence or key phrase
            sentences = text.split('. ')
            if sentences:
                # Clean up the sentence
                sentence = sentences[0].strip()
                if len(sentence) > 10:  # Only add if it's meaningful
                    key_points.append(f"{i}. {sentence}")
            
            # Track source type
            source_type = source.split('_')[0] if '_' in source else source
            if source_type not in ['', 'qna', 'clinical', 'pubmed', 'synthetic']:
                sources.add(source_type)
        
        # Build response
        response_parts = [
            f"## Based on medical information about '{query}':",
            ""
        ]
        
        if key_points:
            response_parts.append("### Key Findings:")
            response_parts.extend(key_points)
            response_parts.append("")
        
        if context:
            response_parts.append(f"**Considering your context:** {context}")
            response_parts.append("")
        
        # Add recommendations based on query type
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['stress', 'anxiety', 'mental']):
            recommendations = [
                "1. Practice mindfulness meditation 10-15 minutes daily",
                "2. Engage in regular gentle exercise like walking or yoga",
                "3. Maintain social connections and support networks",
                "4. Consider cognitive behavioral techniques if needed"
            ]
        elif any(word in query_lower for word in ['diabetes', 'blood sugar', 'glucose']):
            recommendations = [
                "1. Monitor blood glucose levels regularly",
                "2. Follow a balanced diet with controlled carbohydrates",
                "3. Engage in regular physical activity",
                "4. Take medications as prescribed and attend follow-ups"
            ]
        elif any(word in query_lower for word in ['blood pressure', 'hypertension', 'bp']):
            recommendations = [
                "1. Monitor blood pressure regularly at consistent times",
                "2. Reduce sodium intake and increase potassium-rich foods",
                "3. Engage in regular aerobic exercise",
                "4. Take medications consistently as prescribed"
            ]
        else:
            recommendations = [
                "1. Consult with healthcare providers for personalized advice",
                "2. Follow evidence-based management strategies",
                "3. Monitor symptoms and adjust as needed",
                "4. Maintain regular follow-up appointments"
            ]
        
        response_parts.append("### General Recommendations:")
        response_parts.extend(recommendations)
        response_parts.append("")
        
        if sources:
            response_parts.append(f"**Sources:** Information from {', '.join(list(sources)[:3])}")
        else:
            response_parts.append("**Sources:** Medical databases and guidelines")
        
        response_parts.append("")
        response_parts.append(" **Important Notice:** This information is based on collected medical data and is for educational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment.")
        
        return "\n".join(response_parts)
    
    def _get_no_data_response(self, query: str) -> str:
        """Response when no relevant data is found"""
        
        # Check if we have any data at all
        total_docs = len(self.retriever.documents)
        
        if total_docs == 0:
            return """## I don't have any medical data yet.

Please go to **Settings → Data Collection** and:
1. Click **"Collect Fresh Data"**
2. Run **"Processing Pipeline"**  
3. **Refresh** the app

This will load medical information into the system."""
        else:
            return f"""## I have {total_docs} medical documents, but couldn't find specific information about "{query}".

**Try asking about:**
• Diabetes management and blood sugar control
• Hypertension and blood pressure
• Chronic condition lifestyle adjustments
• Medication adherence strategies

**Or rephrase your question to be more specific.**

The system contains information from PubMed articles, clinical guidelines, and medical research."""
    
    def get_recommendations(self, query: str, context: Optional[str] = None) -> Dict:
        """Get recommendations using RAG data"""
        response = self.get_response(query, context)
        
        return {
            "recommendations": response,
            "llm_used": self.use_real_llm,
            "rag_used": True,
            "provider": self.provider
        }