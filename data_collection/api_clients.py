# data_collection/api_clients.py
"""
API Clients for Medical Data Sources
Access PubMed, ClinicalTrials.gov, and other medical APIs
"""

import requests
import json
import logging
from typing import Dict, List, Optional
import xml.etree.ElementTree as ET
from datetime import datetime
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PubMedClient:
    """Client for PubMed/PubMed Central API"""
    
    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    
    def __init__(self, email: str = "your_email@example.com"):
        """
        Initialize PubMed client
        
        Args:
            email: Email for API usage tracking (required by NCBI)
        """
        self.email = email
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': f'HealthResearchBot/1.0 ({email})'
        })
    
    def search_articles(self, query: str, max_results: int = 50) -> List[Dict]:
        """
        Search PubMed for articles
        
        Args:
            query: Search query (e.g., "diabetes treatment")
            max_results: Maximum number of results
            
        Returns:
            List of article metadata
        """
        try:
            # Search for article IDs
            search_url = f"{self.BASE_URL}/esearch.fcgi"
            params = {
                'db': 'pubmed',
                'term': query,
                'retmax': max_results,
                'retmode': 'json',
                'email': self.email
            }
            
            response = self.session.get(search_url, params=params, timeout=15)
            response.raise_for_status()
            
            search_data = response.json()
            article_ids = search_data.get('esearchresult', {}).get('idlist', [])
            
            if not article_ids:
                logger.warning(f"No articles found for query: {query}")
                return []
            
            # Fetch article details
            articles = self._fetch_article_details(article_ids)
            return articles
            
        except Exception as e:
            logger.error(f"Error searching PubMed: {e}")
            return []
    
    def _fetch_article_details(self, article_ids: List[str]) -> List[Dict]:
        """Fetch details for multiple articles"""
        if not article_ids:
            return []
        
        try:
            fetch_url = f"{self.BASE_URL}/efetch.fcgi"
            params = {
                'db': 'pubmed',
                'id': ','.join(article_ids),
                'retmode': 'xml',
                'email': self.email
            }
            
            response = self.session.get(fetch_url, params=params, timeout=30)
            response.raise_for_status()
            
            # Parse XML response
            root = ET.fromstring(response.content)
            articles = []
            
            for article in root.findall('.//PubmedArticle'):
                article_data = self._parse_article_xml(article)
                if article_data:
                    articles.append(article_data)
            
            return articles
            
        except Exception as e:
            logger.error(f"Error fetching article details: {e}")
            return []
    
    def _parse_article_xml(self, article_element) -> Optional[Dict]:
        """Parse PubMed XML article element"""
        try:
            # Extract basic information
            medline = article_element.find('.//MedlineCitation')
            if medline is None:
                return None
            
            # Article title
            title_elem = medline.find('.//ArticleTitle')
            title = title_elem.text if title_elem is not None else "No Title"
            
            # Abstract
            abstract_elem = medline.find('.//AbstractText')
            abstract = abstract_elem.text if abstract_elem is not None else ""
            
            # Authors
            authors = []
            author_list = medline.findall('.//Author')
            for author in author_list[:5]:  # Limit to first 5 authors
                last_name = author.find('LastName')
                fore_name = author.find('ForeName')
                if last_name is not None:
                    author_name = last_name.text
                    if fore_name is not None:
                        author_name = f"{fore_name.text} {author_name}"
                    authors.append(author_name)
            
            # Journal information
            journal_elem = medline.find('.//Journal')
            journal_title = ""
            if journal_elem is not None:
                title_elem = journal_elem.find('Title')
                journal_title = title_elem.text if title_elem is not None else ""
            
            # Publication date
            pub_date = medline.find('.//PubMedPubDate[@PubStatus="pubmed"]')
            year = ""
            if pub_date is not None:
                year_elem = pub_date.find('Year')
                year = year_elem.text if year_elem is not None else ""
            
            # PubMed ID
            pmid_elem = medline.find('.//PMID')
            pmid = pmid_elem.text if pmid_elem is not None else ""
            
            # Keywords
            keywords = []
            keyword_list = medline.findall('.//Keyword')
            for kw in keyword_list:
                if kw.text:
                    keywords.append(kw.text)
            
            return {
                'pmid': pmid,
                'title': title,
                'abstract': abstract,
                'authors': authors,
                'journal': journal_title,
                'year': year,
                'keywords': keywords,
                'source': 'pubmed',
                'collected_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error parsing article XML: {e}")
            return None


class ClinicalTrialsClient:
    """Client for ClinicalTrials.gov API"""
    
    BASE_URL = "https://clinicaltrials.gov/api/v2/studies"
    
    def search_trials(self, condition: str, max_results: int = 30) -> List[Dict]:
        """
        Search for clinical trials
        
        Args:
            condition: Medical condition (e.g., "diabetes")
            max_results: Maximum results
            
        Returns:
            List of trial information
        """
        try:
            params = {
                'format': 'json',
                'query.cond': condition,
                'pageSize': max_results,
                'fields': 'NCTId,BriefTitle,Condition,InterventionName,Phase,BriefSummary'
            }
            
            response = requests.get(self.BASE_URL, params=params, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            trials = []
            
            for study in data.get('studies', [])[:max_results]:
                protocol_section = study.get('protocolSection', {})
                
                trial_data = {
                    'nct_id': study.get('protocolSection', {}).get('identificationModule', {}).get('nctId', ''),
                    'title': protocol_section.get('identificationModule', {}).get('briefTitle', ''),
                    'conditions': protocol_section.get('conditionsModule', {}).get('conditions', []),
                    'interventions': self._extract_interventions(protocol_section),
                    'phase': protocol_section.get('designModule', {}).get('phases', ['N/A'])[0],
                    'summary': protocol_section.get('descriptionModule', {}).get('briefSummary', ''),
                    'source': 'clinicaltrials.gov',
                    'collected_at': datetime.now().isoformat()
                }
                
                trials.append(trial_data)
            
            return trials
            
        except Exception as e:
            logger.error(f"Error fetching clinical trials: {e}")
            return []
    
    def _extract_interventions(self, protocol_section: Dict) -> List[str]:
        """Extract intervention names from trial data"""
        interventions = []
        interventions_module = protocol_section.get('armsInterventionsModule', {})
        
        for intervention in interventions_module.get('interventions', []):
            name = intervention.get('name', '')
            if name:
                interventions.append(name)
        
        return interventions


class HealthGovClient:
    """Client for various health.gov APIs"""
    
    def __init__(self):
        self.cache = {}
    
    def get_cdc_guidelines(self, disease: str) -> List[Dict]:
        """
        Get CDC guidelines (simulated - would need actual API access)
        
        Args:
            disease: Disease name
            
        Returns:
            List of guideline documents
        """
        # Note: CDC doesn't have a simple public API for all guidelines
        # This is a simulated version - in reality you'd need to scrape
        
        cdc_simulated_data = {
            'diabetes': [
                {
                    'title': 'Standards of Medical Care in Diabetes',
                    'content': 'The American Diabetes Association publishes annual standards covering all aspects of diabetes care.',
                    'year': '2024',
                    'source': 'CDC/ADA Guidelines'
                }
            ],
            'hypertension': [
                {
                    'title': '2017 ACC/AHA Guideline for High Blood Pressure',
                    'content': 'Guidelines for prevention, detection, evaluation, and management of high blood pressure.',
                    'year': '2017',
                    'source': 'ACC/AHA Guidelines'
                }
            ]
        }
        
        return cdc_simulated_data.get(disease.lower(), [])