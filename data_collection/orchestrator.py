# data_collection/orchestrator.py
"""
Main Data Collection Orchestrator
Coordinates web scraping, API calls, and synthetic generation
"""

import json
import asyncio
import logging
from pathlib import Path
from typing import Dict, List
from datetime import datetime
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from data_collection.scraper import HealthcareScraper
from data_collection.api_clients import PubMedClient, ClinicalTrialsClient
from data_collection.synthetic_generator import SyntheticDataGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataCollectionOrchestrator:
    """Orchestrates all data collection activities"""
    
    def __init__(self, config_path: str = "data_collection/sources.json"):
        """
        Initialize the orchestrator
        
        Args:
            config_path: Path to data sources configuration
        """
        self.config_path = Path(config_path)
        self.data_dir = Path("data")
        self.raw_dir = self.data_dir / "collected" / "raw"
        self.processed_dir = self.data_dir / "collected" / "processed"
        
        # Create directories
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize clients
        self.pubmed_client = PubMedClient()
        self.trials_client = ClinicalTrialsClient()
        self.synthetic_gen = SyntheticDataGenerator()  # Add API key if available
    
    def _load_config(self) -> Dict:
        """Load configuration file with better error handling"""
        try:
            if not self.config_path.exists():
                logger.warning(f"Config file not found: {self.config_path}. Creating default.")
                return self._create_default_config()
            
            with open(self.config_path, 'r') as f:
                content = f.read().strip()
                if not content:
                    logger.warning("Config file is empty. Creating default configuration.")
                    return self._create_default_config()
                
                return json.loads(content)
                
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in config file: {e}")
            logger.info("Creating default configuration instead.")
            return self._create_default_config()
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return self._create_default_config()
    
    def _create_default_config(self) -> Dict:
        """Create default configuration"""
        default_config = {
            "web_sources": [
                {
                    "name": "CDC Chronic Conditions",
                    "url": "https://www.cdc.gov/chronicdisease/about/index.htm",
                    "type": "html"
                },
                {
                    "name": "WHO Noncommunicable Diseases",
                    "url": "https://www.who.int/news-room/fact-sheets/detail/noncommunicable-diseases",
                    "type": "html"
                }
            ],
            "synthetic_prompts": [
                "Generate a comprehensive guide on managing type 2 diabetes",
                "Create patient education material about hypertension",
                "Explain lifestyle modifications for chronic conditions"
            ]
        }
        
        # Save the default config
        try:
            with open(self.config_path, 'w') as f:
                json.dump(default_config, f, indent=2)
            logger.info(f"Created default config at {self.config_path}")
        except Exception as e:
            logger.error(f"Could not save default config: {e}")
        
        return default_config
    
    async def collect_all_data(self) -> Dict:
        """
        Collect data from all sources
        
        Returns:
            Dictionary with collection statistics
        """
        stats = {
            'start_time': datetime.now().isoformat(),
            'sources': {},
            'total_documents': 0
        }
        
        try:
            # 1. Web Scraping
            web_stats = await self.collect_web_data()
            stats['sources']['web'] = web_stats
            
            # 2. PubMed Articles
            pubmed_stats = self.collect_pubmed_data()
            stats['sources']['pubmed'] = pubmed_stats
            
            # 3. Clinical Trials
            trials_stats = self.collect_clinical_trials()
            stats['sources']['clinical_trials'] = trials_stats
            
            # 4. Synthetic Data
            synthetic_stats = self.generate_synthetic_data()
            stats['sources']['synthetic'] = synthetic_stats
            
            # Calculate totals
            for source_stats in stats['sources'].values():
                stats['total_documents'] += source_stats.get('documents_collected', 0)
            
            stats['end_time'] = datetime.now().isoformat()
            stats['success'] = True
            
            # Save collection report
            self._save_collection_report(stats)
            
            logger.info(f"Data collection complete. Total documents: {stats['total_documents']}")
            return stats
            
        except Exception as e:
            logger.error(f"Error in data collection: {e}")
            stats['error'] = str(e)
            stats['success'] = False
            return stats
    
    async def collect_web_data(self) -> Dict:
        """Collect data from web sources"""
        logger.info("Starting web data collection...")
        
        stats = {
            'sources_scraped': 0,
            'documents_collected': 0,
            'errors': 0
        }
        
        web_sources = self.config.get('web_sources', [])
        collected_data = []
        
        async with HealthcareScraper(rate_limit=2.0) as scraper:
            for source in web_sources[:5]:  # Limit to 5 sources for demo
                try:
                    if source['type'] == 'html':
                        # Scrape single page
                        data = scraper.scrape_single_page(
                            source['url'], 
                            source.get('selectors')
                        )
                        
                        if 'error' not in data:
                            collected_data.append(data)
                            stats['documents_collected'] += 1
                            
                            # Save individual file
                            self._save_raw_data(data, source_type='web')
                        else:
                            stats['errors'] += 1
                    
                    stats['sources_scraped'] += 1
                    
                except Exception as e:
                    logger.error(f"Error scraping {source.get('name')}: {e}")
                    stats['errors'] += 1
        
        # Save batch
        if collected_data:
            batch_file = self.raw_dir / f"web_collection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(batch_file, 'w') as f:
                json.dump(collected_data, f, indent=2)
        
        return stats
    
    def collect_pubmed_data(self) -> Dict:
        """Collect data from PubMed"""
        logger.info("Collecting PubMed articles...")
        
        stats = {
            'searches_performed': 0,
            'articles_found': 0,
            'documents_collected': 0
        }
        
        # Search for chronic condition articles
        chronic_conditions = [
            "diabetes mellitus",
            "hypertension",
            "chronic obstructive pulmonary disease",
            "arthritis",
            "asthma",
            "heart failure",
            "chronic kidney disease"
        ]
        
        all_articles = []
        
        for condition in chronic_conditions[:3]:  # Limit for demo
            try:
                articles = self.pubmed_client.search_articles(
                    query=f"{condition} treatment guidelines",
                    max_results=15
                )
                
                if articles:
                    all_articles.extend(articles)
                    stats['articles_found'] += len(articles)
                    
                    # Save individual articles
                    for article in articles:
                        self._save_raw_data(article, source_type='pubmed')
                
                stats['searches_performed'] += 1
                logger.info(f"Found {len(articles)} articles for {condition}")
                
            except Exception as e:
                logger.error(f"Error searching PubMed for {condition}: {e}")
        
        # Save batch
        if all_articles:
            batch_file = self.raw_dir / f"pubmed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(batch_file, 'w') as f:
                json.dump(all_articles, f, indent=2)
            
            stats['documents_collected'] = len(all_articles)
        
        return stats
    
    def collect_clinical_trials(self) -> Dict:
        """Collect clinical trial data"""
        logger.info("Collecting clinical trial data...")
        
        stats = {
            'conditions_searched': 0,
            'trials_found': 0,
            'documents_collected': 0
        }
        
        conditions = ["diabetes", "hypertension", "arthritis", "asthma"]
        all_trials = []
        
        for condition in conditions:
            try:
                trials = self.trials_client.search_trials(
                    condition=condition,
                    max_results=10
                )
                
                if trials:
                    all_trials.extend(trials)
                    stats['trials_found'] += len(trials)
                    
                    # Save individual trials
                    for trial in trials:
                        self._save_raw_data(trial, source_type='clinical_trial')
                
                stats['conditions_searched'] += 1
                
            except Exception as e:
                logger.error(f"Error searching trials for {condition}: {e}")
        
        # Save batch
        if all_trials:
            batch_file = self.raw_dir / f"trials_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(batch_file, 'w') as f:
                json.dump(all_trials, f, indent=2)
            
            stats['documents_collected'] = len(all_trials)
        
        return stats
    
    def generate_synthetic_data(self) -> Dict:
        """Generate synthetic medical data"""
        logger.info("Generating synthetic data...")
        
        stats = {
            'topics_covered': 0,
            'documents_generated': 0,
            'qna_pairs_generated': 0
        }
        
        # Topics from config or default
        synthetic_prompts = self.config.get('synthetic_prompts', [])
        topics = ["diabetes", "hypertension", "arthritis", "asthma", "heart disease"]
        
        if synthetic_prompts:
            # Extract topics from prompts
            topics = list(set(topics + [p.split()[-1] for p in synthetic_prompts if len(p.split()) > 2]))
        
        # Generate documents
        documents = self.synthetic_gen.generate_medical_content(
            topics=topics[:5],  # Limit topics
            num_documents=25  # Target 25 synthetic documents
        )
        
        if documents:
            # Save documents
            batch_file = self.raw_dir / f"synthetic_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(batch_file, 'w') as f:
                json.dump(documents, f, indent=2)
            
            stats['documents_generated'] = len(documents)
            stats['topics_covered'] = len(set(doc['topic'] for doc in documents))
            
            # Save individual files
            for doc in documents:
                self._save_raw_data(doc, source_type='synthetic')
        
        # Generate Q&A pairs
        qna_data = []
        for topic in topics[:3]:  # Limit topics for Q&A
            qna_pairs = self.synthetic_gen.generate_qna_pairs(
                topic=topic,
                num_pairs=5
            )
            qna_data.extend(qna_pairs)
        
        if qna_data:
            qna_file = self.raw_dir / f"qna_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(qna_file, 'w') as f:
                json.dump(qna_data, f, indent=2)
            
            stats['qna_pairs_generated'] = len(qna_data)
        
        return stats
    
    def _save_raw_data(self, data: Dict, source_type: str):
        """Save individual raw data document"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        doc_id = data.get('doc_id', data.get('pmid', data.get('nct_id', f"doc_{timestamp}")))
        
        filename = f"{source_type}_{doc_id}.json"
        filepath = self.raw_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _save_collection_report(self, stats: Dict):
        """Save collection statistics report"""
        report_file = self.processed_dir / f"collection_report_{datetime.now().strftime('%Y%m%d')}.json"
        
        with open(report_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Collection report saved: {report_file}")
    
    def get_collection_summary(self) -> str:
        """Get a human-readable summary of collected data"""
        # Count files in raw directory
        raw_files = list(self.raw_dir.glob("*.json"))
        
        by_source = {}
        for file in raw_files:
            source = file.name.split('_')[0]
            by_source[source] = by_source.get(source, 0) + 1
        
        summary = f"Data Collection Summary ({len(raw_files)} total documents):\n"
        for source, count in by_source.items():
            summary += f"  - {source}: {count} documents\n"
        
        return summary


def main():
    """Main entry point for data collection"""
    logger.info("Starting Chronic Condition Data Collection Pipeline")
    
    orchestrator = DataCollectionOrchestrator()
    
    try:
        # Run async collection
        stats = asyncio.run(orchestrator.collect_all_data())
        
        if stats.get('success'):
            summary = orchestrator.get_collection_summary()
            logger.info("\n" + summary)
            
            # Print detailed stats
            print("\n=== Data Collection Complete ===")
            print(f"Total documents collected: {stats['total_documents']}")
            print(f"Time taken: {stats['start_time']} to {stats['end_time']}")
            
            for source, source_stats in stats['sources'].items():
                print(f"\n{source.upper()}:")
                for key, value in source_stats.items():
                    print(f"  {key}: {value}")
        else:
            logger.error(f"Data collection failed: {stats.get('error')}")
            
    except KeyboardInterrupt:
        logger.info("Data collection interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()