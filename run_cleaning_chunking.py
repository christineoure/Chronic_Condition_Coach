# run_cleaning_chunking.py
"""
Clean, process, and chunk the collected data for embedding
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import re
from typing import List, Dict, Any
import logging
from datetime import datetime
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataCleaner:
    """Clean and preprocess collected medical data"""
    
    def __init__(self):
        self.data_dir = Path("data")
        self.raw_dir = self.data_dir / "collected" / "raw"
        self.cleaned_dir = self.data_dir / "cleaned"
        self.cleaned_dir.mkdir(exist_ok=True)
        
    def load_all_collected_data(self) -> List[Dict]:
        """Load all collected JSON files"""
        all_data = []
        json_files = list(self.raw_dir.glob("*.json"))
        
        logger.info(f"Found {len(json_files)} JSON files to process")
        
        for file_path in json_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    
                # Handle different data structures
                if isinstance(data, list):
                    all_data.extend(data)
                elif isinstance(data, dict):
                    # Check if it's a batch file or single document
                    if 'source_url' in data or 'content' in data or 'abstract' in data:
                        all_data.append(data)
                    elif 'studies' in data:  # Clinical trials batch
                        studies = data.get('studies', [])
                        for study in studies:
                            all_data.append(study)
                    else:
                        # Try to extract documents from dict values
                        for key, value in data.items():
                            if isinstance(value, list):
                                all_data.extend(value)
                else:
                    logger.warning(f"Unknown data format in {file_path.name}")
                    
            except Exception as e:
                logger.error(f"Error loading {file_path.name}: {e}")
        
        logger.info(f"Loaded {len(all_data)} documents")
        return all_data
    
    def extract_text_content(self, document: Dict) -> str:
        """Extract text content from different document formats"""
        text_parts = []
        
        # PubMed articles
        if 'abstract' in document and document['abstract']:
            text_parts.append(document['abstract'])
        
        if 'title' in document and document['title']:
            text_parts.append(document['title'])
        
        # Web scraped content
        if 'content' in document and document['content']:
            text_parts.append(document['content'])
        
        # Synthetic content
        if 'text' in document and document['text']:
            text_parts.append(document['text'])
        
        # Clinical trials
        if 'summary' in document and document['summary']:
            text_parts.append(document['summary'])
        
        if 'briefTitle' in document and document['briefTitle']:
            text_parts.append(document['briefTitle'])
        
        # Q&A pairs
        if 'question' in document and document['question']:
            text_parts.append(f"Q: {document['question']}")
        if 'answer' in document and document['answer']:
            text_parts.append(f"A: {document['answer']}")
        
        # Join all parts
        full_text = " ".join([str(part) for part in text_parts if part])
        
        return full_text.strip()
    
    def extract_metadata(self, document: Dict) -> Dict:
        """Extract metadata from document"""
        metadata = {
            'source_type': 'unknown',
            'topic': 'general',
            'date': datetime.now().strftime('%Y-%m-%d')
        }
        
        # Determine source type
        if 'source' in document:
            metadata['source_type'] = document['source']
        elif 'source_url' in document:
            metadata['source_type'] = 'web'
        elif 'pmid' in document:
            metadata['source_type'] = 'pubmed'
        elif 'nct_id' in document:
            metadata['source_type'] = 'clinical_trial'
        
        # Extract topic/keywords
        if 'topic' in document:
            metadata['topic'] = document['topic']
        elif 'conditions' in document and document['conditions']:
            metadata['topic'] = document['conditions'][0] if isinstance(document['conditions'], list) else document['conditions']
        elif 'keywords' in document and document['keywords']:
            if isinstance(document['keywords'], list) and document['keywords']:
                metadata['topic'] = document['keywords'][0]
            elif document['keywords']:
                metadata['topic'] = document['keywords']
        
        # Add document ID if available
        if 'pmid' in document:
            metadata['doc_id'] = f"pubmed_{document['pmid']}"
        elif 'nct_id' in document:
            metadata['doc_id'] = f"trial_{document['nct_id']}"
        elif 'doc_id' in document:
            metadata['doc_id'] = document['doc_id']
        else:
            import hashlib
            content = self.extract_text_content(document)[:100]
            metadata['doc_id'] = hashlib.md5(content.encode()).hexdigest()[:10]
        
        return metadata
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?;:\-\'\"()\[\]]', ' ', text)
        
        # Remove multiple periods
        text = re.sub(r'\.{2,}', '.', text)
        
        # Normalize quotes
        text = text.replace('"', "'").replace('`', "'").replace('´', "'")
        
        # Remove URLs
        text = re.sub(r'https?://\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove reference patterns like [1], [2-5]
        text = re.sub(r'\[\d+(?:-\d+)?\]', '', text)
        
        # Trim
        text = text.strip()
        
        return text
    
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks"""
        if not text:
            return []
        
        words = text.split()
        chunks = []
        
        if len(words) <= chunk_size:
            chunks.append(text)
        else:
            start = 0
            while start < len(words):
                end = start + chunk_size
                chunk = " ".join(words[start:end])
                chunks.append(chunk)
                start += chunk_size - overlap
        
        return chunks
    
    def process_documents(self) -> pd.DataFrame:
        """Process all documents into clean chunks"""
        logger.info("Loading collected data...")
        documents = self.load_all_collected_data()
        
        if not documents:
            logger.warning("No documents found to process!")
            return pd.DataFrame()
        
        logger.info(f"Processing {len(documents)} documents...")
        
        chunks_data = []
        doc_count = 0
        
        for doc in documents:
            try:
                # Extract and clean text
                raw_text = self.extract_text_content(doc)
                if not raw_text or len(raw_text.strip()) < 50:
                    continue  # Skip very short documents
                
                clean_text = self.clean_text(raw_text)
                
                # Extract metadata
                metadata = self.extract_metadata(doc)
                
                # Create chunks
                text_chunks = self.chunk_text(clean_text, chunk_size=400, overlap=50)
                
                for i, chunk in enumerate(text_chunks):
                    if len(chunk.split()) < 20:  # Skip very short chunks
                        continue
                    
                    chunk_data = {
                        'chunk_id': f"{metadata.get('doc_id', 'doc')}_{i}",
                        'text': chunk,
                        'source_type': metadata['source_type'],
                        'topic': metadata['topic'],
                        'document_index': doc_count,
                        'chunk_index': i,
                        'total_chunks': len(text_chunks),
                        'word_count': len(chunk.split()),
                        'original_doc_id': metadata.get('doc_id', 'unknown')
                    }
                    
                    # Add source-specific metadata
                    if 'title' in doc:
                        chunk_data['title'] = str(doc['title'])[:200]
                    if 'year' in doc:
                        chunk_data['year'] = doc['year']
                    if 'authors' in doc and doc['authors']:
                        chunk_data['authors'] = ", ".join(doc['authors'][:3]) if isinstance(doc['authors'], list) else str(doc['authors'])
                    
                    chunks_data.append(chunk_data)
                
                doc_count += 1
                if doc_count % 10 == 0:
                    logger.info(f"Processed {doc_count}/{len(documents)} documents...")
                    
            except Exception as e:
                logger.error(f"Error processing document: {e}")
                continue
        
        logger.info(f"Created {len(chunks_data)} chunks from {doc_count} documents")
        
        # Create DataFrame
        df = pd.DataFrame(chunks_data)
        
        # Save to files
        self.save_results(df)
        
        return df
    
    def save_results(self, df: pd.DataFrame):
        """Save cleaned chunks to files"""
        # Save as JSON
        json_path = self.cleaned_dir / "chunks.json"
        df.to_json(json_path, orient='records', indent=2)
        logger.info(f"Saved chunks to {json_path}")
        
        # Save as CSV
        csv_path = self.cleaned_dir / "chunks.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved chunks to {csv_path}")
        
        # Save summary
        summary = {
            'total_chunks': len(df),
            'total_documents_processed': df['document_index'].nunique(),
            'source_distribution': df['source_type'].value_counts().to_dict(),
            'topic_distribution': df['topic'].value_counts().head(10).to_dict(),
            'average_chunk_size': df['word_count'].mean(),
            'processing_date': datetime.now().isoformat()
        }
        
        summary_path = self.cleaned_dir / "processing_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Saved processing summary to {summary_path}")
        
        # Print summary
        print("\n" + "="*50)
        print(" DATA PROCESSING SUMMARY")
        print("="*50)
        print(f"Total chunks created: {len(df):,}")
        print(f"Documents processed: {df['document_index'].nunique():,}")
        print(f"Average chunk size: {df['word_count'].mean():.1f} words")
        print("\nSources:")
        for source, count in summary['source_distribution'].items():
            print(f"  - {source}: {count:,} chunks")
        print("\nTop topics:")
        for topic, count in list(summary['topic_distribution'].items())[:5]:
            print(f"  - {topic}: {count:,} chunks")


def main():
    """Main processing function"""
    print("🧹 Starting data cleaning and chunking pipeline...")
    
    cleaner = DataCleaner()
    
    try:
        df = cleaner.process_documents()
        
        if len(df) > 0:
            print("\n Processing complete!")
            print(f"\nNext step: Run embeddings_vector_db.py to create vector embeddings")
        else:
            print("\n No data was processed. Check your collected data.")
            
    except Exception as e:
        logger.error(f"Error in processing pipeline: {e}")
        raise


if __name__ == "__main__":
    main()