# Now let's update test_full_system.py to use the correct function names

#!/usr/bin/env python
"""
Test the complete Chronic Condition RAG System
"""

import sys
import os

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_all_components():
    """Test all components of the system."""
    print("=" * 70)
    print(" TESTING CHRONIC CONDITION RAG SYSTEM")
    print("=" * 70)
    
    print("\n SYSTEM STATUS CHECK:")
    print("-" * 40)
    
    # Check if vector_db exists
    if os.path.exists("vector_db"):
        print(" Vector database: Found")
        if os.path.exists("vector_db/collection_info.json"):
            with open("vector_db/collection_info.json", "r") as f:
                import json
                info = json.load(f)
                print(f"   Collection: {info['collection_name']}")
                print(f"   Chunks: {info['total_chunks']}")
                print(f"   Sources: {len(info['sources'])}")
    else:
        print(" Vector database: Not found")
        return
    
    # Check data files
    if os.path.exists("data/cleaned/chunks.json"):
        with open("data/cleaned/chunks.json", "r") as f:
            import json
            chunks = json.load(f)
            print(f" Cleaned chunks: {len(chunks)}")
    else:
        print(" Cleaned chunks: Not found")
        return
    
    # Test 1: Retriever
    print("\n TESTING RETRIEVER:")
    print("-" * 40)
    try:
        from src.retriever import test_retriever
        retriever = test_retriever()
        if retriever:
            print(" Retriever: Working")
        else:
            print(" Retriever: Failed")
            return
    except Exception as e:
        print(f" Retriever Error: {e}")
        return
    
    # Test 2: RAG Pipeline
    print("\n\n TESTING RAG PIPELINE:")
    print("-" * 40)
    try:
        from src.rag_pipeline import test_rag_pipeline
        pipeline = test_rag_pipeline()
        print(" RAG Pipeline: Working")
    except Exception as e:
        print(f" RAG Pipeline Error: {e}")
        return
    
    # Test 3: Coach Agent
    print("\n\n TESTING COACH AGENT:")
    print("-" * 40)
    try:
        from src.coach_agent import test_coach_agent
        coach = test_coach_agent()
        print(" Coach Agent: Working")
    except Exception as e:
        print(f" Coach Agent Error: {e}")
        return
    
    print("\n" + "=" * 70)
    print(" ALL TESTS COMPLETE! SYSTEM IS OPERATIONAL")
    print("=" * 70)
    
    # Demo
    print("\n🎯 QUICK DEMO:")
    print("-" * 40)
    
    from src.coach_agent import ChronicConditionCoach
    coach = ChronicConditionCoach()
    
    demo_queries = [
        "best foods for diabetes",
        "how to lower blood pressure naturally",
        "managing stress with chronic condition"
    ]
    
    for query in demo_queries[:1]:  # Just show first one
        result = coach.quick_advice(query)
        print(f"\nQuery: '{query}'")
        print(f"Sources used: {result['sources']}")
        print("\nTop recommendation:")
        print(result['advice'][:200] + "...")
        break
    
    print("\n📁 PROJECT STRUCTURE:")
    print("""
chronic_condition_rag/
├── data/                    # Health knowledge documents
├── vector_db/              # ChromaDB vector database  
├── src/                    # Source code
│   ├── retriever.py       # Vector DB queries
│   ├── rag_pipeline.py    # RAG generation
│   └── coach_agent.py     # Main agent
├── requirements.txt
├── README.md
└── test_full_system.py    # This test script
""")

if __name__ == "__main__":
    test_all_components()


