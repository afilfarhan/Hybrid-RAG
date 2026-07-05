"""Web-based data connector for ingestion pipeline."""

from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
import asyncio
import aiohttp
from pathlib import Path

logger = logging.getLogger(__name__)


class WebConnector:
    """Connector for web-based data sources."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize web connector.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.base_urls = config.get('base_urls', [])
        self.metadata = config.get('metadata', {})
        self.timeout = config.get('timeout', 30)
        self.max_pages = config.get('max_pages', 100)
        self.rate_limit_delay = config.get('rate_limit_delay', 0.5)
        
    async def _fetch_page(self, session: aiohttp.ClientSession, url: str) -> Optional[str]:
        """Fetch a single page.
        
        Args:
            session: aiohttp session
            url: URL to fetch
            
        Returns:
            Page content or None if failed
        """
        try:
            async with session.get(url, timeout=self.timeout) as response:
                if response.status == 200:
                    return await response.text()
                else:
                    logger.warning(f"Failed to fetch {url}: HTTP {response.status}")
                    return None
        except Exception as e:
            logger.error(f"Error fetching {url}: {e}")
            return None
    
    async def _extract_links(self, html: str, base_url: str) -> List[str]:
        """Extract links from HTML content.
        
        Args:
            html: HTML content
            base_url: Base URL for relative links
            
        Returns:
            List of extracted URLs
        """
        from bs4 import BeautifulSoup
        
        soup = BeautifulSoup(html, 'html.parser')
        links = []
        
        for a in soup.find_all('a', href=True):
            href = a['href']
            if href.startswith('http'):
                links.append(href)
            elif href.startswith('/'):
                from urllib.parse import urljoin
                links.append(urljoin(base_url, href))
                
        return links[:self.max_pages]
    
    async def _extract_text(self, html: str) -> str:
        """Extract text content from HTML.
        
        Args:
            html: HTML content
            
        Returns:
            Extracted text
        """
        from bs4 import BeautifulSoup
        
        soup = BeautifulSoup(html, 'html.parser')
        
        for script in soup(['script', 'style', 'nav', 'footer', 'header']):
            script.decompose()
            
        text = soup.get_text()
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        
        return "\n".join(lines)
    
    async def ingest_url(self, url: str) -> Dict[str, Any]:
        """Ingest a single URL.
        
        Args:
            url: URL to ingest
            
        Returns:
            Ingestion result
        """
        import hashlib
        
        async with aiohttp.ClientSession() as session:
            html = await self._fetch_page(session, url)
            
            if not html:
                return {
                    'doc_id': None,
                    'content': None,
                    'metadata': None,
                    'status': 'failed',
                    'error': 'Failed to fetch page'
                }
            
            content = await self._extract_text(html)
            content_hash = hashlib.md5(content.encode()).hexdigest()[:16]
            
            metadata = {
                'source_type': 'web',
                'url': url,
                'last_crawled': datetime.now().isoformat(),
                **self.metadata
            }
            
            return {
                'doc_id': f"web_{content_hash}",
                'content': content,
                'metadata': metadata,
                'status': 'success'
            }
    
    async def ingest_directory(self, base_url: str) -> List[Dict[str, Any]]:
        """Ingest all pages from a directory/website.
        
        Args:
            base_url: Base URL to start from
            
        Returns:
            List of ingestion results
        """
        results = []
        visited = set()
        to_visit = [base_url]
        
        async with aiohttp.ClientSession() as session:
            while to_visit and len(visited) < self.max_pages:
                url = to_visit.pop(0)
                
                if url in visited:
                    continue
                    
                visited.add(url)
                
                html = await self._fetch_page(session, url)
                
                if not html:
                    continue
                
                content = await self._extract_text(html)
                content_hash = hashlib.md5(content.encode()).hexdigest()[:16]
                
                result = {
                    'doc_id': f"web_{content_hash}",
                    'content': content,
                    'metadata': {
                        'source_type': 'web',
                        'url': url,
                        'last_crawled': datetime.now().isoformat(),
                        **self.metadata
                    },
                    'status': 'success'
                }
                
                results.append(result)
                
                # Extract and add new links
                links = await self._extract_links(html, url)
                for link in links:
                    if link not in visited and link.startswith(base_url):
                        to_visit.append(link)
                
                # Rate limiting
                await asyncio.sleep(self.rate_limit_delay)
        
        return results
    
    async def ingest_all(self) -> List[Dict[str, Any]]:
        """Ingest all configured URLs.
        
        Returns:
            List of ingestion results
        """
        all_results = []
        
        for url in self.base_urls:
            logger.info(f"Ingesting from: {url}")
            results = await self.ingest_directory(url)
            all_results.extend(results)
            
        return all_results
