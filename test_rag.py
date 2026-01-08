# test_rag_fixed.py
#!/usr/bin/env python
"""
Test the RAG system with your collected data
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.simple_llm import SimpleLLMClient

def test_rag():
    print(" Testing RAG System with Your Collected Data")
    print("=" * 50)
    
    # Initialize RAG client
    client = SimpleLLMClient(provider="rag")
    
    # Test queries
    test_queries = [
        "Managing stress with chronic condition?",
        "Best foods for diabetes?",
        "How to lower blood pressure naturally?",
        "Exercise for arthritis patients?"
    ]
    
    for query in test_queries:
        print(f"\n Query: '{query}'")
        print("-" * 30)
        
        try:
            response = client.get_response(query)
            
            # Check if we get a real response
            if "install openai package" in response.lower():
                print(" Still getting mock response!")
                print("Response preview:", response[:100])
            elif "don't have any medical data" in response.lower():
                print(" No data found!")
                print("Make sure you have run data collection and processing.")
            else:
                print(" Real RAG response generated!")
                print(f"Response length: {len(response)} characters")
                print(f"\n Response preview:")
                print("-" * 30)
                print(response[:300] + "...")
                print("-" * 30)
                
        except Exception as e:
            print(f" Error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 50)
    print(" Checking data availability...")
    
    data_dir = Path("data/collected/raw")
    if data_dir.exists():
        files = list(data_dir.glob("*.json"))
        print(f"Collected files: {len(files)}")
        
        # Check file types
        from collections import Counter
        file_types = Counter([f.name.split('_')[0] for f in files])
        print("File types:", dict(file_types))
    else:
        print(" No collected data directory found!")
    
    cleaned_dir = Path("data/cleaned")
    if cleaned_dir.exists() and (cleaned_dir / "chunks.json").exists():
        import json
        with open(cleaned_dir / "chunks.json", 'r') as f:
            chunks = json.load(f)
        print(f" Cleaned data exists: {len(chunks)} chunks")
    else:
        print(" No cleaned data - run cleaning pipeline")

if __name__ == "__main__":
    test_rag()