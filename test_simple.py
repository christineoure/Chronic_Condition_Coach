#!/usr/bin/env python
"""
Simple test of the system
"""

print(" Testing Chronic Condition Coach System...")

# Test brain
print("\n1. Testing Brain...")
from src.brain_with_llm import LLMEnhancedBrain
brain = LLMEnhancedBrain()
print(f"   Memory count: {len(brain.memory)}")
print(f"   LLM available: {brain.get_brain_stats()['llm_available']}")

# Test recommendations
print("\n2. Testing Recommendations...")
result = brain.get_recommendations("How to manage diabetes with poor sleep?")
print(f"   Memory used: {result['memory_context_used']}")
print(f"   LLM used: {result['llm_used']}")
print(f"   Sample: {result['recommendations'][:150]}...")

# Test insights
print("\n3. Testing Insights...")
test_session = {
    "user_query": "diabetes sleep management",
    "detected_triggers": ["poor sleep", "irregular schedule"],
    "summary": "Sleep affecting glucose"
}
insights = brain.generate_insights(test_session)
print(f"   Pattern insights: {len(insights['pattern_insights'])}")
print(f"   Memory insights: {len(insights['memory_based_insights'])}")

print("\n System is working!")
print("\n To start the UI, run:")
print("   streamlit run app_simple.py")

