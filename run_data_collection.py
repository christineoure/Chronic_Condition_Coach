#!/usr/bin/env python
"""
Standalone script to run data collection
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from data_collection.orchestrator import DataCollectionOrchestrator
import json

async def main():
    print(" Starting Chronic Condition Data Collection")
    print("=" * 50)
    
    orchestrator = DataCollectionOrchestrator()
    
    print("\n Running full data collection pipeline...")
    stats = await orchestrator.collect_all_data()
    
    if stats.get('success'):
        print("\n" + "=" * 50)
        print(" Data collection complete!")
        print(f"Total documents: {stats.get('total_documents', 0)}")
        
        summary = orchestrator.get_collection_summary()
        print(summary)
        
        # Save the summary
        with open("data_collection_summary.txt", "w") as f:
            f.write(summary)
        
        print("\nNext steps:")
        print("1. Process data: python run_cleaning_chunking.py")
        print("2. Create embeddings: python embeddings_vector_db.py")
        print("3. Start app: streamlit run app_simple.py")
    else:
        print(f"\n Data collection failed: {stats.get('error')}")

if __name__ == "__main__":
    asyncio.run(main())