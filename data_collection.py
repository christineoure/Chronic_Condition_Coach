# run_data_collection.py
"""
Standalone script to run data collection
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from data_collection.orchestrator import DataCollectionOrchestrator
from data_collection.scraper import HealthcareScraper
import json

async def main():
    print(" Starting Chronic Condition Data Collection")
    print("=" * 50)
    
    orchestrator = DataCollectionOrchestrator()
    
    print("\n1.  Collecting web data...")
    web_stats = await orchestrator.collect_web_data()
    print(f"   Collected: {web_stats.get('documents_collected', 0)} documents")
    
    print("\n2.  Fetching PubMed articles...")
    pubmed_stats = orchestrator.collect_pubmed_data()
    print(f"   Collected: {pubmed_stats.get('documents_collected', 0)} articles")
    
    print("\n3.  Gathering clinical trials...")
    trial_stats = orchestrator.collect_clinical_trials()
    print(f"   Collected: {trial_stats.get('documents_collected', 0)} trials")
    
    print("\n4.  Generating synthetic data...")
    synthetic_stats = orchestrator.generate_synthetic_data()
    print(f"   Generated: {synthetic_stats.get('documents_generated', 0)} documents")
    print(f"   Q&A pairs: {synthetic_stats.get('qna_pairs_generated', 0)}")
    
    print("\n" + "=" * 50)
    summary = orchestrator.get_collection_summary()
    print(summary)
    
    # Save the summary
    with open("data_collection_summary.txt", "w") as f:
        f.write(summary)
    
    print("\n Data collection complete!")
    print("\nNext steps:")
    print("1. Run your cleaning pipeline: python run_cleaning_chunking.py")
    print("2. Generate embeddings: python embeddings_vector_db.py")
    print("3. Restart your Streamlit app: streamlit run app_simple.py")

if __name__ == "__main__":
    asyncio.run(main())