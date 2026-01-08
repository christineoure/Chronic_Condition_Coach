# Now update the coach_agent.py to fix any import issues

#!/usr/bin/env python
"""
Coach Agent Module
Orchestrates the complete chronic condition coaching system.
"""

import json
import pandas as pd
from datetime import datetime, timedelta
from src.rag_pipeline import RAGPipeline

class ChronicConditionCoach:
    def __init__(self):
        """Initialize the complete coaching agent."""
        print("Initializing Chronic Condition Coach...")
        
        # Initialize components
        self.pipeline = RAGPipeline()
        print("RAG Pipeline loaded")
        
        # Initialize pattern database
        self.patterns_db = self._load_patterns()
        print("Pattern database loaded")
        
        print("Coach Agent ready!")
    
    def _load_patterns(self):
        """Load known health patterns and triggers."""
        return {
            "sleep_deprivation": {
                "triggers": ["<6 hours sleep for 3+ nights", "irregular sleep schedule"],
                "effects": ["elevated resting HR", "increased sugar cravings", "higher glucose"],
                "recommendations": ["sleep consistency", "bedtime routine", "light management"]
            },
            "chronic_stress": {
                "triggers": ["low HRV for 5+ days", "high workload", "poor recovery"],
                "effects": ["elevated cortisol", "blood pressure spikes", "immune suppression"],
                "recommendations": ["stress recovery", "boundary setting", "mindfulness"]
            }
        }
    
    def analyze_wearable_data(self, wearable_data=None):
        """Analyze wearable data for patterns (mock implementation)."""
        if wearable_data is None:
            # Generate mock wearable data
            dates = pd.date_range(end=datetime.now(), periods=7).strftime('%Y-%m-%d').tolist()
            wearable_data = pd.DataFrame({
                'date': dates,
                'steps': [8200, 9100, 4300, 10200, 7500, 6800, 8900],
                'sleep_hours': [6.1, 5.4, 4.9, 7.2, 6.8, 5.9, 7.0],
                'resting_hr': [72, 75, 78, 68, 71, 74, 70],
                'hrv': [48, 42, 38, 55, 50, 45, 52]
            })
        
        insights = []
        
        # Analyze sleep patterns
        if wearable_data['sleep_hours'].mean() < 6.5:
            insights.append({
                "type": "sleep_deprivation",
                "severity": "moderate",
                "description": f"Average sleep: {wearable_data['sleep_hours'].mean():.1f}h (target: 7-8h)",
                "recommendation": "Focus on sleep extension and consistency"
            })
        
        # Analyze stress patterns
        if wearable_data['hrv'].mean() < 45:
            insights.append({
                "type": "chronic_stress",
                "severity": "moderate",
                "description": f"Low HRV: {wearable_data['hrv'].mean():.0f} ms (target: >50 ms)",
                "recommendation": "Prioritize stress recovery techniques"
            })
        
        return {
            "insights": insights,
            "summary_stats": {
                "avg_steps": int(wearable_data['steps'].mean()),
                "avg_sleep": wearable_data['sleep_hours'].mean(),
                "avg_hrv": wearable_data['hrv'].mean(),
                "data_days": len(wearable_data)
            }
        }
    
    def detect_triggers(self, wearable_analysis):
        """Detect specific triggers from wearable analysis."""
        triggers = []
        
        for insight in wearable_analysis["insights"]:
            pattern_type = insight["type"]
            if pattern_type in self.patterns_db:
                pattern = self.patterns_db[pattern_type]
                triggers.extend(pattern["triggers"])
        
        return list(set(triggers))  # Remove duplicates
    
    def coach_session(self, user_query, wearable_data=None, user_context=""):
        """Complete coaching session with analysis and recommendations."""
        print(f"\n{'='*60}")
        print(f"COACHING SESSION")
        print(f"{'='*60}")
        
        # Step 1: Analyze wearable data (if provided)
        print("\nStep 1: Analyzing patterns...")
        wearable_analysis = self.analyze_wearable_data(wearable_data)
        
        if wearable_analysis["insights"]:
            print("   Detected patterns:")
            for insight in wearable_analysis["insights"]:
                print(f"   • {insight['description']} ({insight['severity']})")
        else:
            print("   No significant patterns detected")
        
        # Step 2: Detect triggers
        print("\n Step 2: Identifying triggers...")
        triggers = self.detect_triggers(wearable_analysis)
        if triggers:
            print("   Potential triggers:")
            for trigger in triggers[:3]:  # Show top 3
                print(f"   • {trigger}")
        else:
            print("   No specific triggers identified")
        
        # Step 3: Generate recommendations using RAG
        print("\n Step 3: Generating personalized recommendations...")
        
        # Enhance query with detected patterns
        enhanced_query = user_query
        if wearable_analysis["insights"]:
            pattern_desc = ", ".join([i["type"].replace("_", " ") for i in wearable_analysis["insights"]])
            enhanced_query = f"{user_query} (Detected patterns: {pattern_desc})"
        
        # Generate recommendations
        result = self.pipeline.generate_recommendation(
            user_query=enhanced_query
        )

        
        # Step 4: Compile complete response
        response = {
            "session_id": f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "user_query": user_query,
            "wearable_analysis": wearable_analysis,
            "detected_triggers": triggers,
            "recommendations": result["recommendations"],
            "sources_used": result["sources_used"],
            "summary": self._create_summary(wearable_analysis, triggers)
        }
        
        return response
    
    def _create_summary(self, wearable_analysis, triggers):
        """Create a concise summary of findings."""
        summary_parts = []
        
        # Add wearable summary
        stats = wearable_analysis["summary_stats"]
        summary_parts.append(
            f"Analysis of {stats['data_days']} days: "
            f"{stats['avg_steps']:,} avg steps, "
            f"{stats['avg_sleep']:.1f}h avg sleep"
        )
        
        # Add pattern summary
        if wearable_analysis["insights"]:
            patterns = [i["type"].replace("_", " ") for i in wearable_analysis["insights"]]
            summary_parts.append(f"Patterns: {', '.join(patterns)}")
        
        return ". ".join(summary_parts) + "."
    
    def quick_advice(self, query):
        """Quick advice without full analysis."""
        result = self.pipeline.generate_recommendation(query)
        return {
            "advice": result["recommendations"],
            "sources": len(result["sources_used"])
        }

def test_coach_agent():
    """Test the complete coaching agent."""
    print(" Testing Chronic Condition Coach...")
    
    coach = ChronicConditionCoach()
    
    # Test quick advice
    print(f"\n{'='*60}")
    print(" Quick Advice Test")
    print(f"{'='*60}")
    quick = coach.quick_advice("How to manage blood sugar with poor sleep?")
    print(f"\nAdvice sources: {quick['sources']}")
    print("\n Advice:")
    print(quick['advice'][:400] + "...")
    
    # Test full session
    print(f"\n{'='*60}")
    print(" Full Coaching Session Test")
    print(f"{'='*60}")
    result = coach.coach_session(
        user_query="Managing blood sugar with inconsistent sleep",
        user_context="I work night shifts and my glucose is high in mornings"
    )
    
    print(f"\n Summary: {result['summary']}")
    print(f"\n Triggers identified: {len(result['detected_triggers'])}")
    print(f" Sources referenced: {len(result['sources_used'])}")
    
    return coach

if __name__ == "__main__":
    test_coach_agent()

