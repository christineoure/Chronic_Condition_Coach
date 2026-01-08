# collect_75_documents.py
#!/usr/bin/env python
"""
Targeted data collection to ensure 75+ authoritative medical documents
"""

import asyncio
import json
from pathlib import Path
from data_collection.orchestrator import DataCollectionOrchestrator
from data_collection.scraper import HealthcareScraper
from data_collection.api_clients import PubMedClient, ClinicalTrialsClient
from data_collection.synthetic_generator import SyntheticDataGenerator
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def collect_targeted_data():
    """Collect data with specific targets for 75+ documents"""
    
    print(" Targeted Data Collection for 75+ Medical Documents")
    print("=" * 60)
    
    # Load enhanced config
    with open("data_collection/sources.json", 'r') as f:
        config = json.load(f)
    
    # Create directories
    data_dir = Path("data/collected/raw")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    total_documents = 0
    collected_data = []
    
    # 1. Collect from key authoritative websites (20+ documents)
    print("\n1.  Collecting from authoritative medical websites...")
    key_sites = [
        "https://www.cdc.gov/chronicdisease/about/index.htm",
        "https://www.who.int/news-room/fact-sheets/detail/noncommunicable-diseases",
        "https://www.heart.org/en/health-topics/high-blood-pressure",
        "https://www.diabetes.org/diabetes",
        "https://www.niddk.nih.gov/health-information/diabetes",
        "https://www.mayoclinic.org/diseases-conditions"
    ]
    
    async with HealthcareScraper(rate_limit=3.0) as scraper:
        for url in key_sites:
            try:
                data = scraper.scrape_single_page(url)
                if 'error' not in data:
                    collected_data.append(data)
                    total_documents += 1
                    print(f"    {url.split('/')[2]}: Collected")
                else:
                    print(f"    {url.split('/')[2]}: Failed")
            except Exception as e:
                print(f"    Error: {e}")
    
    # 2. Collect from PubMed (30+ documents)
    print("\n2.  Collecting PubMed articles...")
    pubmed = PubMedClient()
    
    medical_queries = [
        "chronic disease self management lifestyle intervention",
        "diabetes prevention nutrition exercise",
        "hypertension management diet physical activity",
        "arthritis pain management exercise",
        "chronic obstructive pulmonary disease pulmonary rehabilitation"
    ]
    
    for query in medical_queries:
        try:
            articles = pubmed.search_articles(query, max_results=10)
            collected_data.extend(articles)
            total_documents += len(articles)
            print(f"    '{query}': {len(articles)} articles")
        except Exception as e:
            print(f"    Error on '{query}': {e}")
    
    # 3. Generate synthetic lifestyle guidance (25+ documents)
    print("\n3.  Generating synthetic lifestyle guidance...")
    synth_gen = SyntheticDataGenerator()
    
    lifestyle_topics = [
        "diabetes", "hypertension", "arthritis", "asthma", 
        "chronic pain", "heart failure", "COPD", "kidney disease"
    ]
    
    synthetic_docs = synth_gen.generate_medical_content(
        topics=lifestyle_topics,
        num_documents=25
    )
    
    collected_data.extend(synthetic_docs)
    total_documents += len(synthetic_docs)
    print(f"    Generated {len(synthetic_docs)} synthetic documents")
    
    # 4. Save all collected data
    print("\n4.  Saving collected data...")
    timestamp = Path("data").mkdir(exist_ok=True)
    
    # Save as single file
    output_file = data_dir / f"medical_collection_{Path('').joinpath('*').stem}.json"
    with open(output_file, 'w') as f:
        json.dump(collected_data, f, indent=2)
    
    # Also save individual documents
    for i, doc in enumerate(collected_data):
        doc_file = data_dir / f"doc_{i}.json"
        with open(doc_file, 'w') as f:
            json.dump(doc, f, indent=2)
    
    print(f"\n COLLECTION COMPLETE!")
    print(f" Total documents collected: {total_documents}")
    print(f" Saved to: {output_file}")
    
    if total_documents >= 75:
        print(" SUCCESS: Exceeded target of 75 documents!")
    else:
        print(f" WARNING: Only collected {total_documents}/75 documents")
        print("   Consider running the full orchestrator for more data.")
    
    return total_documents

if __name__ == "__main__":
    asyncio.run(collect_targeted_data())