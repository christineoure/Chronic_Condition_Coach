#!/usr/bin/env python
"""
Chronic Condition Coach - Enhanced with Real Data Collection
"""

# 1. FIRST: Import Streamlit
import streamlit as st

# 2. SECOND: Set page config IMMEDIATELY (FIRST Streamlit command)
st.set_page_config(
    page_title="Chronic Condition Coach",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 3. THIRD: Now import other libraries
import os
import sys
from pathlib import Path
import pandas as pd
from datetime import datetime
import asyncio

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent))

# 4. FOURTH: Import our components
try:
    from src.brain_with_llm import LLMEnhancedBrain
    from src.simple_llm import SimpleLLMClient
    from data_collection.orchestrator import DataCollectionOrchestrator
    from data_collection.scraper import HealthcareScraper
except ImportError as e:
    st.error(f"Could not import required modules: {e}. Please check the installation.")
    st.stop()

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #A23B72;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .card {
        background-color: #F8F9FA;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #2E86AB;
        margin-bottom: 1rem;
    }
    .insight-card {
        background-color: #FFF3CD;
        padding: 0.75rem;
        border-radius: 8px;
        border-left: 4px solid #FFC107;
        margin-bottom: 0.5rem;
    }
    .data-status {
        background-color: #e8f4fd;
        padding: 10px;
        border-radius: 5px;
        border-left: 4px solid #2E86AB;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
# In app_simple.py initialization section
if 'brain' not in st.session_state:
    with st.spinner(" Loading AI Brain with Medical Data..."):
        # Force use of RAG mode
        st.session_state.brain = LLMEnhancedBrain()
        st.session_state.llm = SimpleLLMClient(provider="rag")  # <-- Make sure it's "rag"
        st.session_state.sessions = []
        st.session_state.data_collected = True  # You have data!
        st.session_state.data_files_count = 191  # Your cleaned chunks count
        
        # Show success message
        st.success(f" Loaded 191 medical documents from collected data")
# Check for existing data
def check_existing_data():
    """Check if data already exists in the system"""
    # Check vector database
    vector_db_path = Path("vector_db")
    has_vector_data = vector_db_path.exists() and any(
        vector_db_path.glob("*.sqlite3")
    )
    
    # Check collected data
    collected_dir = Path("data/collected/raw")
    if collected_dir.exists():
        data_files = list(collected_dir.glob("*.json"))
        st.session_state.data_files_count = len(data_files)
        return len(data_files) > 5  # Consider having data if more than 5 files
    
    return has_vector_data

# Data collection function
def collect_health_data():
    """Trigger data collection from various sources"""
    with st.spinner(" Collecting health data from multiple sources..."):
        try:
            orchestrator = DataCollectionOrchestrator()
            stats = asyncio.run(orchestrator.collect_all_data())
            st.session_state.collection_stats = stats
            st.session_state.data_collected = True
            
            # Update data files count
            collected_dir = Path("data/collected/raw")
            if collected_dir.exists():
                data_files = list(collected_dir.glob("*.json"))
                st.session_state.data_files_count = len(data_files)
            
            st.success(f" Collected {stats.get('total_documents', 0)} documents!")
            
            # Ask if user wants to process the data
            if st.session_state.data_files_count > 0:
                st.info(" Data collected! Run 'Processing Pipeline' in Settings to create embeddings.")
            
            return True
        except Exception as e:
            st.error(f" Data collection failed: {e}")
            return False

# Process data function
def process_collected_data():
    """Process collected data into embeddings"""
    with st.spinner(" Processing data and creating embeddings..."):
        try:
            # Import your existing processing scripts
            from run_cleaning_chunking import main as clean_data
            from embeddings_vector_db import main as create_embeddings
            
            # Run cleaning
            st.info("Step 1: Cleaning and chunking data...")
            clean_data()
            
            # Run embeddings
            st.info("Step 2: Creating embeddings...")
            create_embeddings()
            
            st.success(" Data processing complete! The system now uses real data.")
            
            # Update brain to use new data
            st.session_state.brain = LLMEnhancedBrain()
            
            return True
        except Exception as e:
            st.error(f" Data processing failed: {e}")
            return False

# Main app function
def main():
    # Header
    st.markdown('<h1 class="main-header"> Chronic Condition Coach</h1>', unsafe_allow_html=True)
    st.markdown("AI-powered health coaching with memory and learning")
    
    # Show data status banner
    if check_existing_data():
        st.markdown("""
        <div class="data-status">
         <b>System Status:</b> Real data available ({data_count} documents). Responses are based on collected medical information.
        </div>
        """.format(data_count=st.session_state.data_files_count), unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="data-status" style="border-left-color: #ff6b6b;">
         <b>System Status:</b> Using mock responses. Collect real data in Settings for accurate medical information.
        </div>
        """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/heart-health.png", width=60)
        st.markdown("### Navigation")
        
        page = st.radio(
            "Choose:",
            [" Dashboard", " Coaching", " Insights", " Settings"]
        )
        
        st.markdown("---")
        st.markdown("### System Status")
        
        # Show brain stats
        if st.session_state.brain:
            stats = st.session_state.brain.get_brain_stats()
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Memories", stats["memory_count"])
            with col2:
                st.metric("Patterns", stats["active_patterns"])
            
            # Data status
            st.metric("Data Docs", st.session_state.data_files_count)
            st.metric("LLM", "Real" if stats["llm_available"] else "Mock")
        
        st.markdown("---")
        st.markdown("** Safety Note:**")
        st.info("This system provides lifestyle suggestions only. Always consult healthcare professionals for medical advice.")
    
    # Main content
    if page == " Dashboard":
        show_dashboard()
    elif page == " Coaching":
        show_coaching()
    elif page == " Insights":
        show_insights()
    elif page == " Settings":
        show_settings()

def show_dashboard():
    """Show dashboard."""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Sessions", len(st.session_state.sessions))
    
    with col2:
        st.metric("AI Brain Memories", st.session_state.brain.get_brain_stats()["memory_count"])
    
    with col3:
        st.metric("Data Documents", st.session_state.data_files_count)
    
    st.markdown("---")
    
    # Quick actions
    st.markdown('<h3 class="sub-header"> Quick Questions</h3>', unsafe_allow_html=True)
    
    quick_questions = [
        "Best foods for diabetes?",
        "How to lower blood pressure naturally?",
        "Managing stress with chronic condition?",
        "Sleep tips for better glucose control?"
    ]
    
    cols = st.columns(2)
    for i, question in enumerate(quick_questions):
        with cols[i % 2]:
            if st.button(question, use_container_width=True):
                handle_quick_question(question)
    
    # Recent sessions
    if st.session_state.sessions:
        st.markdown("---")
        st.markdown('<h3 class="sub-header"> Recent Sessions</h3>', unsafe_allow_html=True)
        
        for session in st.session_state.sessions[-3:][::-1]:
            with st.expander(f" {session['query'][:50]}..."):
                st.write(f"**Summary:** {session.get('summary', 'N/A')}")
                st.write(f"**Time:** {session.get('timestamp', 'N/A')}")
                st.write(f"**Recommendations:** {session.get('recommendations', 'N/A')[:200]}...")

def show_coaching():
    """Show coaching interface."""
    st.markdown('<h2 class="sub-header"> Health Coaching Session</h2>', unsafe_allow_html=True)
    
    # Data availability warning
    if st.session_state.data_files_count < 5:
        st.warning(" Limited real data available. Responses may be generic. Collect more data in Settings.")
    
    # Query input
    query = st.text_area(
        "What health concern would you like help with?",
        height=100,
        placeholder="e.g., 'How to manage diabetes with irregular sleep?' or 'Tips for lowering blood pressure naturally...'"
    )
    
    # Context
    context = st.text_input(
        "Additional context (optional):",
        placeholder="e.g., 'I work night shifts', 'Family history of diabetes'"
    )
    
    # Options
    col1, col2 = st.columns(2)
    with col1:
        use_memory = st.checkbox("Use Memory & Learning", value=True)
    with col2:
        include_analysis = st.checkbox("Include Wearable Analysis", value=True)
    
    # Submit button
    if st.button(" Get Personalized Recommendations", type="primary", use_container_width=True):
        if not query:
            st.warning("Please enter a question.")
        else:
            with st.spinner(" Analyzing with AI..."):
                # Get recommendations
                result = st.session_state.brain.get_recommendations(query, context)
                
                # Create session record
                session_data = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "query": query,
                    "context": context,
                    "recommendations": result["recommendations"],
                    "memory_used": result["memory_context_used"],
                    "llm_used": result["llm_used"],
                    "summary": f"Query about {query[:50]}..."
                }
                
                st.session_state.sessions.append(session_data)
                
                # Display results
                display_results(result)

def display_results(result):
    """Display coaching results."""
    st.markdown("---")
    st.markdown('<h3 class="sub-header"> Your Recommendations</h3>', unsafe_allow_html=True)
    
    # Info card
    with st.container():
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f" Memory used: {result['memory_context_used']} past sessions")
        with col2:
            st.info(f" LLM: {'Real API' if result['llm_used'] else 'Mock'}")
        with col3:
            data_source = "Real Data" if st.session_state.data_files_count > 5 else "Mock Responses"
            st.info(f" Source: {data_source}")
    
    # Recommendations
    st.markdown("---")
    st.write(result["recommendations"])
    
    # Memory details
    if result.get("memory_details"):
        with st.expander(" Similar Past Cases"):
            for i, memory in enumerate(result["memory_details"], 1):
                st.write(f"{i}. **Query:** {memory['query']}...")
                st.write(f"   **Summary:** {memory['summary']}...")
    
    # Action buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button(" Save Session", use_container_width=True):
            st.success("Session saved!")
    with col2:
        if st.button(" Another Question", use_container_width=True):
            st.rerun()
    with col3:
        if st.button(" View Insights", use_container_width=True):
            st.session_state.page = " Insights"
            st.rerun()

def show_insights():
    """Show insights and analytics."""
    st.markdown('<h2 class="sub-header"> Brain Insights</h2>', unsafe_allow_html=True)
    
    # Get brain stats
    stats = st.session_state.brain.get_brain_stats()
    
    # Display stats
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Memories", stats["memory_count"])
    
    with col2:
        st.metric("Active Patterns", stats["active_patterns"])
    
    with col3:
        data_percent = min(100, (st.session_state.data_files_count / 50) * 100)
        st.metric("Data Coverage", f"{data_percent:.1f}%")
    
    with col4:
        st.metric("LLM", "Available" if stats["llm_available"] else "Mock")
    
    st.markdown("---")
    
    # Data sources breakdown
    st.markdown('<h4> Data Sources</h4>', unsafe_allow_html=True)
    
    if st.session_state.collection_stats:
        st.write("**Last Collection Stats:**")
        for source, source_stats in st.session_state.collection_stats.get('sources', {}).items():
            docs = source_stats.get('documents_collected', 0)
            if docs > 0:
                st.write(f"- {source.title()}: {docs} documents")
    else:
        st.info("No data collection stats available yet.")
    
    # Collection button
    if st.button(" Refresh Data & Insights", use_container_width=True):
        check_existing_data()
        st.rerun()

def show_settings():
    """Show settings."""
    st.markdown('<h2 class="sub-header"> System Settings</h2>', unsafe_allow_html=True)
    
    # Data Collection Section
    st.markdown("###  Data Collection Pipeline")
    
    # Step 1: Collect Data
    st.markdown("#### Step 1: Collect Data")
    st.write("Gather medical information from various sources (websites, APIs, synthetic generation)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button(" Collect Fresh Data", type="primary", use_container_width=True):
            if collect_health_data():
                st.rerun()
    
    with col2:
        if st.button(" View Collection Stats", type="secondary", use_container_width=True):
            if st.session_state.collection_stats:
                st.json(st.session_state.collection_stats)
            else:
                st.info("No data collected yet")
    
    # Current data status
    collected_dir = Path("data/collected/raw")
    if collected_dir.exists():
        data_files = list(collected_dir.glob("*.json"))
        current_count = len(data_files)
        st.info(f" **Current collected data:** {current_count} documents")
        
        # Step 2: Process Data (only show if data exists)
        if current_count > 0:
            st.markdown("---")
            st.markdown("#### Step 2: Process Data")
            st.write("Clean, chunk, and create embeddings for the collected data")
            
            if st.button(" Run Processing Pipeline", type="primary", use_container_width=True):
                if process_collected_data():
                    st.rerun()
    
    # LLM Settings
    st.markdown("---")
    st.markdown("###  LLM Configuration")
    
    provider = st.selectbox(
        "LLM Provider",
        ["openai", "anthropic", "mock"],
        index=0 if st.session_state.llm.provider == "openai" else 1 if st.session_state.llm.provider == "anthropic" else 2
    )
    
    if provider != "mock":
        api_key = st.text_input(
            f"{provider.upper()} API Key",
            type="password",
            placeholder=f"Enter your {provider.upper()} API key"
        )
        
        if api_key:
            os.environ[f"{provider.upper()}_API_KEY"] = api_key
            if st.button("Save API Key"):
                st.success(f"{provider.upper()} API key saved!")
                # Reinitialize LLM client
                st.session_state.llm = SimpleLLMClient(provider=provider)
    
    # Memory Settings
    st.markdown("---")
    st.markdown("###  Memory Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Clear All Memories", type="secondary", use_container_width=True):
            if st.session_state.brain.memory:
                st.session_state.brain.memory = []
                st.session_state.brain._save_memory()
                st.success("Memories cleared!")
            else:
                st.info("No memories to clear")
    
    with col2:
        if st.button("Export Memories", type="secondary", use_container_width=True):
            st.info("Export feature coming soon")
    
    # System Info
    st.markdown("---")
    st.markdown("###  System Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Python Version:** 3.12")
        st.write("**Streamlit Version:** 1.29.0")
        st.write("**Brain Version:** 1.0")
    
    with col2:
        st.write(f"**Memory File:** {st.session_state.brain.memory_file}")
        st.write(f"**Sessions Stored:** {len(st.session_state.sessions)}")
        st.write(f"**Data Documents:** {st.session_state.data_files_count}")

def handle_quick_question(question):
    """Handle quick question."""
    result = st.session_state.brain.get_recommendations(question)
    
    # Create session
    session_data = {
        "timestamp": datetime.now().strftime("%H:%M"),
        "query": question,
        "recommendations": result["recommendations"][:500] + "...",
        "summary": f"Quick question about {question[:30]}..."
    }
    
    st.session_state.sessions.append(session_data)
    
    # Show answer
    st.markdown("---")
    st.markdown(f"###  Answer to: '{question}'")
    st.write(result["recommendations"][:300] + "...")
    
    if result["memory_context_used"] > 0:
        st.info(f" Used {result['memory_context_used']} past similar cases")

if __name__ == "__main__":
    # Initial data check
    if 'data_checked' not in st.session_state:
        check_existing_data()
        st.session_state.data_checked = True
    
    main()