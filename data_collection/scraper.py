# data_collection/scraper.py
"""
Web Scraper for Healthcare Data Collection
Scrapes medical websites for chronic condition information
"""

import asyncio
import aiohttp
from bs4 import BeautifulSoup
import requests
from urllib.parse import urljoin, urlparse
import json
import logging
from typing import List, Dict, Optional
import re
import time
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HealthcareScraper:
    """Scraper for collecting healthcare information from medical websites"""
    
    def __init__(self, rate_limit: float = 1.0, max_concurrent: int = 3):
        """
        Initialize the healthcare scraper
        
        Args:
            rate_limit: Seconds to wait between requests
            max_concurrent: Maximum concurrent requests
        """
        self.rate_limit = rate_limit
        self.max_concurrent = max_concurrent
        self.session = None
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (compatible; HealthResearchBot/1.0; +http://example.com)'
        }
        
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(headers=self.headers)
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    def scrape_single_page(self, url: str, selectors: Optional[Dict] = None) -> Dict:
        """
        Scrape a single webpage for healthcare content
        
        Args:
            url: URL to scrape
            selectors: CSS selectors for content extraction
            
        Returns:
            Dictionary containing scraped content
        """
        try:
            logger.info(f"Scraping: {url}")
            
            # Send request with timeout
            response = requests.get(
                url, 
                headers=self.headers, 
                timeout=10,
                verify=True  # SSL verification
            )
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove unwanted elements
            for element in soup.select('script, style, nav, footer, iframe, .advertisement'):
                element.decompose()
            
            # Extract content based on selectors or default heuristics
            content = self._extract_medical_content(soup, selectors)
            
            # Clean and structure the data
            cleaned_data = {
                'source_url': url,
                'title': content.get('title', ''),
                'content': content.get('text', ''),
                'metadata': {
                    'scraped_at': datetime.now().isoformat(),
                    'content_type': 'medical_article',
                    'word_count': len(content.get('text', '').split()),
                    'headers_found': len(content.get('headers', []))
                },
                'headers': content.get('headers', []),
                'keywords': self._extract_keywords(content.get('text', ''))
            }
            
            logger.info(f"Successfully scraped {url} - {cleaned_data['metadata']['word_count']} words")
            return cleaned_data
            
        except Exception as e:
            logger.error(f"Error scraping {url}: {str(e)}")
            return {
                'source_url': url,
                'error': str(e),
                'scraped_at': datetime.now().isoformat()
            }
    
    async def scrape_multiple_pages(self, urls: List[str]) -> List[Dict]:
        """
        Asynchronously scrape multiple pages
        
        Args:
            urls: List of URLs to scrape
            
        Returns:
            List of scraped content dictionaries
        """
        if not self.session:
            async with aiohttp.ClientSession(headers=self.headers) as session:
                self.session = session
                return await self._scrape_concurrent(urls)
        return await self._scrape_concurrent(urls)
    
    async def _scrape_concurrent(self, urls: List[str]) -> List[Dict]:
        """Concurrent scraping implementation"""
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def fetch_with_semaphore(url):
            async with semaphore:
                await asyncio.sleep(self.rate_limit)  # Rate limiting
                return await self._fetch_page(url)
        
        tasks = [fetch_with_semaphore(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions
        valid_results = []
        for result in results:
            if not isinstance(result, Exception):
                valid_results.append(result)
        
        return valid_results
    
    async def _fetch_page(self, url: str) -> Dict:
        """Fetch and parse a single page asynchronously"""
        try:
            async with self.session.get(url, timeout=10) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # Extract content
                    content = self._extract_medical_content(soup)
                    
                    return {
                        'source_url': url,
                        'title': content.get('title', ''),
                        'content': content.get('text', ''),
                        'scraped_at': datetime.now().isoformat(),
                        'status': 'success'
                    }
                else:
                    return {
                        'source_url': url,
                        'error': f'HTTP {response.status}',
                        'scraped_at': datetime.now().isoformat(),
                        'status': 'failed'
                    }
        except Exception as e:
            return {
                'source_url': url,
                'error': str(e),
                'scraped_at': datetime.now().isoformat(),
                'status': 'failed'
            }
    
    def _extract_medical_content(self, soup: BeautifulSoup, selectors: Optional[Dict] = None) -> Dict:
        """
        Extract medical content from BeautifulSoup object
        
        Args:
            soup: BeautifulSoup parsed HTML
            selectors: Custom CSS selectors
            
        Returns:
            Dictionary with title, text, and headers
        """
        # Default selectors for medical content
        default_selectors = {
            'title': 'h1',
            'content': 'article, main, .content, .article-body, #content',
            'headers': 'h1, h2, h3, h4'
        }
        
        selectors = selectors or default_selectors
        
        # Extract title
        title_elem = soup.select_one(selectors['title'])
        title = title_elem.get_text().strip() if title_elem else ''
        
        # Extract main content
        content_selector = selectors.get('content', 'article, main')
        content_elements = soup.select(content_selector)
        
        if not content_elements:
            # Fallback: try to find the largest text container
            all_text = soup.get_text()
            content_text = all_text
        else:
            # Combine all content elements
            content_text = ' '.join([elem.get_text().strip() for elem in content_elements])
        
        # Extract headers for structure
        headers = []
        header_tags = soup.select(selectors.get('headers', 'h1, h2, h3'))
        for header in header_tags:
            headers.append({
                'level': header.name,
                'text': header.get_text().strip()
            })
        
        # Clean up text
        content_text = self._clean_text(content_text)
        
        return {
            'title': title,
            'text': content_text,
            'headers': headers
        }
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove references like [1], [2, 3]
        text = re.sub(r'\[\d+(?:,\s*\d+)*\]', '', text)
        
        # Remove common non-content patterns
        patterns_to_remove = [
            r'Advertisement',
            r'Share this article',
            r'Related:.*',
            r'Read more:.*',
            r'Back to top'
        ]
        
        for pattern in patterns_to_remove:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        return text.strip()
    
    def _extract_keywords(self, text: str, top_n: int = 10) -> List[str]:
        """Extract potential keywords from medical text"""
        # Common medical stopwords (simplified)
        medical_stopwords = {'patient', 'patients', 'study', 'studies', 'research', 
                            'clinical', 'treatment', 'symptoms', 'diagnosis'}
        
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        word_freq = {}
        
        for word in words:
            if word not in medical_stopwords:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Get top N keywords
        sorted_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_keywords[:top_n]]
    
    def discover_related_links(self, base_url: str, depth: int = 1) -> List[str]:
        """
        Discover related pages from a base URL (limited depth)
        
        Args:
            base_url: Starting URL
            depth: How many levels deep to search
            
        Returns:
            List of discovered URLs
        """
        discovered = set()
        
        def _discover(current_url, current_depth):
            if current_depth > depth:
                return
            
            try:
                response = requests.get(current_url, headers=self.headers, timeout=10)
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Find links that look like medical content
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    full_url = urljoin(current_url, href)
                    
                    # Filter for likely medical content
                    if self._is_medical_link(full_url) and full_url not in discovered:
                        discovered.add(full_url)
                        
                        # Recursively discover (with limit)
                        if current_depth < depth:
                            _discover(full_url, current_depth + 1)
                            
            except Exception as e:
                logger.error(f"Error discovering links from {current_url}: {e}")
        
        _discover(base_url, 0)
        return list(discovered)
    
    def _is_medical_link(self, url: str) -> bool:
        """Check if a URL likely points to medical content"""
        medical_keywords = [
            'treatment', 'symptoms', 'diagnosis', 'management',
            'guidelines', 'research', 'study', 'clinical',
            'patient', 'care', 'therapy', 'medication'
        ]
        
        url_lower = url.lower()
        return any(keyword in url_lower for keyword in medical_keywords)