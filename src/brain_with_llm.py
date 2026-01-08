#!/usr/bin/env python
"""
Brain Module with Real LLM Integration - Fixed for Python 3.12
"""

import json
import pickle
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path

# Import our LLM client
try:
    from src.simple_llm import SimpleLLMClient
except ImportError:
    # Create a mock if not available
    class SimpleLLMClient:
        def __init__(self, provider="openai"):
            self.provider = provider
            self.use_real_llm = False
        
        def get_recommendations(self, query: str, context: Optional[str] = None) -> Dict:
            return {
                "recommendations": "Mock LLM response - install openai package for real responses.",
                "llm_used": False
            }

class LLMEnhancedBrain:
    def __init__(self, memory_file="memory/memories.pkl", llm_provider="openai"):
        """Initialize brain with LLM capabilities."""
        print("Initializing LLM-Enhanced Brain...")
        
        # Setup memory
        self.memory_file = memory_file
        self.memory = self._load_memory()
        
        # Initialize LLM client
        try:
            self.llm = SimpleLLMClient(provider=llm_provider)
            print(f"LLM client initialized (Provider: {llm_provider})")
        except Exception as e:
            print(f"Could not initialize LLM client: {e}. Using mock.")
            self.llm = SimpleLLMClient(provider=llm_provider)
        
        # Initialize reasoning
        self.reasoning_patterns = self._initialize_reasoning_patterns()
        
        print(f"Brain loaded with {len(self.memory)} memories")
        print(f"{len(self.reasoning_patterns)} reasoning patterns")
    
    def _load_memory(self):
        """Load memory from disk using built-in pickle."""
        import os
        os.makedirs("memory", exist_ok=True)
        
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Could not load memory: {e}. Starting fresh.")
                return []
        return []
    
    def _save_memory(self):
        """Save memory to disk using built-in pickle."""
        with open(self.memory_file, 'wb') as f:
            pickle.dump(self.memory, f)
    
    def _initialize_reasoning_patterns(self):
        """Initialize reasoning patterns."""
        return {
            "sleep_glucose_connection": {
                "pattern": "sleep < 6 hours → glucose spikes",
                "confidence": 0.85,
                "evidence_count": 0
            },
            "stress_bp_connection": {
                "pattern": "high stress → elevated BP", 
                "confidence": 0.80,
                "evidence_count": 0
            },
            "activity_energy_connection": {
                "pattern": "consistent steps > 7500 → higher energy",
                "confidence": 0.75,
                "evidence_count": 0
            }
        }
    
    def create_memory(self, session_data: Dict) -> str:
        """Create memory entry."""
        query = session_data.get("user_query", "")
        memory_id = hashlib.md5((query + datetime.now().isoformat()).encode()).hexdigest()[:8]
        
        memory_entry = {
            "id": memory_id,
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "patterns": session_data.get("wearable_analysis", {}).get("insights", []),
            "triggers": session_data.get("detected_triggers", []),
            "summary": session_data.get("summary", ""),
            "raw_data": {
                "user_context": session_data.get("user_context", "")[:200],
                "sources_used": [s.get("title", "")[:50] for s in session_data.get("sources_used", [])]
            }
        }
        
        self.memory.append(memory_entry)
        
        # Keep memory manageable (last 100 sessions)
        if len(self.memory) > 100:
            self.memory = self.memory[-100:]
        
        self._save_memory()
        
        # Update reasoning patterns with new evidence
        self._update_patterns_with_evidence(memory_entry)
        
        return memory_id
    
    def _update_patterns_with_evidence(self, memory: Dict):
        """Update reasoning patterns based on new evidence."""
        memory_text = json.dumps(memory).lower()
        
        for pattern_name, pattern in self.reasoning_patterns.items():
            pattern_text = pattern["pattern"].lower()
            keywords = [word for word in pattern_text.split() if len(word) > 3]
            
            # Check if memory contains pattern keywords
            if any(keyword in memory_text for keyword in keywords):
                pattern["evidence_count"] += 1
                # Increase confidence slightly with more evidence
                pattern["confidence"] = min(0.95, pattern["confidence"] + 0.01)
    
    def generate_insights(self, current_session: Dict) -> Dict:
        """Generate insights using available data."""
        # Get similar past memories
        past_memories = self.recall_memories(current_session.get("user_query", ""), limit=3)
        
        insights = {
            "pattern_insights": [],
            "recommendation_insights": [],
            "memory_based_insights": []
        }
        
        # Pattern matching insights
        for pattern_name, pattern in self.reasoning_patterns.items():
            if pattern["confidence"] > 0.7:
                pattern_text = pattern["pattern"].lower()
                session_text = current_session.get("user_query", "").lower()
                
                # Simple keyword matching
                keywords = [word for word in pattern_text.split() if len(word) > 3]
                if any(keyword in session_text for keyword in keywords):
                    insights["pattern_insights"].append(
                        f"Matches pattern: {pattern['pattern']} (confidence: {pattern['confidence']:.2f})"
                    )
        
        # Memory-based insights
        if past_memories:
            insights["memory_based_insights"].append(
                f"Found {len(past_memories)} similar past sessions"
            )
            
            # Check for recurring triggers
            current_triggers = set(current_session.get("detected_triggers", []))
            past_triggers = set()
            
            for memory in past_memories:
                past_triggers.update(set(memory.get("triggers", [])))
            
            common_triggers = current_triggers.intersection(past_triggers)
            if common_triggers:
                insights["recommendation_insights"].append(
                    f"Recurring triggers: {', '.join(list(common_triggers)[:3])}"
                )
        
        # Wearable-based insights
        wearable_insights = current_session.get("wearable_analysis", {}).get("insights", [])
        if wearable_insights:
            primary_insight = wearable_insights[0]
            insights["recommendation_insights"].append(
                f"Focus on: {primary_insight.get('recommendation', 'pattern management')}"
            )
        
        return insights
    
    def recall_memories(self, query: str, limit: int = 5) -> List[Dict]:
        """Recall relevant memories with simple similarity."""
        query_lower = query.lower()
        scored_memories = []
        
        for memory in self.memory[-50:]:  # Check recent memories
            score = 0
            
            # Simple keyword matching
            memory_text = (memory.get("query", "") + " " + memory.get("summary", "")).lower()
            
            query_words = set(query_lower.split())
            memory_words = set(memory_text.split())
            common_words = query_words.intersection(memory_words)
            
            if len(common_words) >= 2:  # At least 2 common words
                score = len(common_words) / max(len(query_words), 1)
            
            if score > 0:
                memory["relevance_score"] = score
                scored_memories.append(memory)
        
        # Sort by relevance
        scored_memories.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        
        return scored_memories[:limit]
    
    def get_recommendations(self, query: str, context: Optional[str] = None) -> Dict:
        """Get recommendations based on collected data."""
        
        # First check if we have collected data
        data_dir = Path("data/collected/raw")
        has_data = data_dir.exists() and any(data_dir.glob("*.json"))
        
        if not has_data:
            # No data collected yet
            return {
                "recommendations": """Please collect medical data first in the Settings page.

Go to Settings → Data Collection and:
1. Click "Collect Fresh Data" 
2. Run "Processing Pipeline"
3. Refresh the app""",
                "memory_context_used": 0,
                "llm_used": False,
                "memory_details": []
            }
        
        # Get memory context
        memory_context = self.recall_memories(query, limit=3)
        
        try:
            # Use the LLM with your collected data
            result = self.llm.get_recommendations(query, context)
            
            return {
                "recommendations": result["recommendations"],
                "memory_context_used": len(memory_context),
                "llm_used": result.get("llm_used", False),
                "memory_details": [
                    {
                        "query": m.get("query", "")[:50],
                        "summary": m.get("summary", "")[:100]
                    }
                    for m in memory_context
                ]
            }
            
        except Exception as e:
            # Fallback to using collected data directly
            return self._get_fallback_from_data(query, memory_context)
    
    def _get_fallback_from_data(self, query: str, memory_context: List[Dict]) -> Dict:
        """Fallback: use collected data directly."""
        # Simple keyword search in collected data
        results = []
        query_words = query.lower().split()
        
        data_dir = Path("data/collected/raw")
        files_checked = 0
        
        for file_path in data_dir.glob("*.json"):
            if files_checked >= 5:  # Limit search to 5 files
                break
                
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict):
                            text = item.get('abstract') or item.get('content') or item.get('text') or item.get('summary') or ''
                            if text and any(word in text.lower() for word in query_words[:3]):
                                results.append({
                                    'text': str(text)[:300] + '...',
                                    'source': file_path.name
                                })
                                if len(results) >= 3:
                                    break
                
                files_checked += 1
                
            except:
                continue
        
        if results:
            response = f"Based on collected medical data about '{query}':\n\n"
            for i, result in enumerate(results, 1):
                response += f"{i}. {result['text']}\n\n"
            response += "Note: This is based on automatically collected medical information. Always consult healthcare professionals."
        else:
            response = f"I have medical data in the system but couldn't find specific information about '{query}'. Try asking about:\n"
            response += "• Diabetes management and nutrition\n"
            response += "• Hypertension and blood pressure control\n"
            response += "• Chronic condition lifestyle modifications\n"
            response += f"\nThe system contains {len(list(data_dir.glob('*.json')))} medical documents."
        
        return {
            "recommendations": response,
            "memory_context_used": len(memory_context),
            "llm_used": False,
            "memory_details": [
                {
                    "query": m.get("query", "")[:50],
                    "summary": m.get("summary", "")[:100]
                }
                for m in memory_context
            ]
        }
    
    def analyze_trends(self) -> Dict:
        """Analyze trends across memories."""
        if len(self.memory) < 3:
            return {"message": "Need more memories for trend analysis"}
        
        # Simple trend analysis
        pattern_counts = {}
        trigger_counts = {}
        
        for memory in self.memory[-20:]:  # Last 20 sessions
            # Count patterns
            for pattern in memory.get("patterns", []):
                pattern_type = pattern.get("type", "")
                pattern_counts[pattern_type] = pattern_counts.get(pattern_type, 0) + 1
            
            # Count triggers
            for trigger in memory.get("triggers", []):
                trigger_counts[trigger] = trigger_counts.get(trigger, 0) + 1
        
        # Find most common
        common_patterns = sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        common_triggers = sorted(trigger_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        
        return {
            "total_sessions_analyzed": len(self.memory[-20:]),
            "common_patterns": common_patterns,
            "common_triggers": common_triggers,
            "time_period": self._get_memory_time_period(self.memory[-20:])
        }
    
    def _get_memory_time_period(self, memories: List[Dict]) -> str:
        """Get time period covered by memories."""
        if not memories:
            return "No memories"
        
        try:
            timestamps = [datetime.fromisoformat(m["timestamp"]) for m in memories]
            oldest = min(timestamps)
            newest = max(timestamps)
            
            days = (newest - oldest).days
            if days == 0:
                return "Today"
            elif days < 7:
                return f"Last {days} days"
            elif days < 30:
                return f"Last {days//7} weeks"
            else:
                return f"Last {days//30} months"
        except:
            return "Unknown period"
    
    def get_brain_stats(self) -> Dict:
        """Get brain statistics."""
        return {
            "memory_count": len(self.memory),
            "reasoning_patterns": len(self.reasoning_patterns),
            "active_patterns": sum(1 for p in self.reasoning_patterns.values() if p["confidence"] > 0.7),
            "avg_pattern_confidence": sum(p["confidence"] for p in self.reasoning_patterns.values()) / len(self.reasoning_patterns) if self.reasoning_patterns else 0,
            "total_evidence": sum(p["evidence_count"] for p in self.reasoning_patterns.values()),
            "llm_available": getattr(self.llm, 'use_real_llm', False)
        }


def test_brain():
    """Test the brain."""
    print("Testing LLM-Enhanced Brain...")
    
    brain = LLMEnhancedBrain(llm_provider="openai")
    
    # Create test session
    test_session = {
        "user_query": "Managing diabetes with poor sleep",
        "user_context": "Sleep 5-6 hours, morning glucose high",
        "wearable_analysis": {
            "insights": [
                {
                    "type": "sleep_deprivation",
                    "severity": "moderate",
                    "description": "Average sleep: 5.5h",
                    "recommendation": "Sleep extension"
                }
            ]
        },
        "detected_triggers": ["<6 hours sleep", "irregular sleep"],
        "summary": "Sleep deprivation affecting glucose control",
        "sources_used": [{"title": "Sleep and Blood Sugar"}]
    }
    
    # Test memory creation
    print("\n1. Creating memory...")
    memory_id = brain.create_memory(test_session)
    print(f"   Memory ID: {memory_id}")
    
    # Test recall
    print("\n2. Testing recall...")
    memories = brain.recall_memories("sleep diabetes", limit=2)
    print(f"   Found {len(memories)} relevant memories")
    
    # Test recommendations
    print("\n3. Getting recommendations...")
    recs = brain.get_recommendations(
        query="How to improve sleep for better blood sugar?",
        context="Night shift worker"
    )
    
    print(f"   Memory context used: {recs['memory_context_used']}")
    print(f"   LLM used: {recs['llm_used']}")
    print(f"\n   Sample:\n   {recs['recommendations'][:200]}...")
    
    # Test insights
    print("\n4. Generating insights...")
    insights = brain.generate_insights(test_session)
    print(f"   Pattern insights: {len(insights['pattern_insights'])}")
    print(f"   Memory insights: {len(insights['memory_based_insights'])}")
    
    # Test stats
    print("\n5. Brain statistics...")
    stats = brain.get_brain_stats()
    print(f"   Memory count: {stats['memory_count']}")
    print(f"   Reasoning patterns: {stats['reasoning_patterns']}")
    print(f"   Active patterns: {stats['active_patterns']}")
    print(f"   LLM available: {stats['llm_available']}")
    
    return brain


if __name__ == "__main__":
    test_brain()