"""API-based data connector for ingestion pipeline."""

from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
import asyncio
import aiohttp
import hashlib

logger = logging.getLogger(__name__)


class APIConnector:
    """Connector for API-based data sources."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize API connector.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.endpoints = config.get('endpoints', [])
        self.headers = config.get('headers', {})
        self.metadata = config.get('metadata', {})
        self.timeout = config.get('timeout', 30)
        self.rate_limit_delay = config.get('rate_limit_delay', 0.5)
        
    async def _fetch_data(self, session: aiohttp.ClientSession, endpoint: str) -> Optional[Dict[str, Any]]:
        """Fetch data from a single endpoint.
        
        Args:
            session: aiohttp session
            endpoint: API endpoint URL
            
        Returns:
            JSON response or None if failed
        """
        try:
            async with session.get(
                endpoint,
                headers=self.headers,
                timeout=self.timeout
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.warning(f"Failed to fetch {endpoint}: HTTP {response.status}")
                    return None
        except Exception as e:
            logger.error(f"Error fetching {endpoint}: {e}")
            return None
    
    async def ingest_endpoint(self, endpoint: str) -> List[Dict[str, Any]]:
        """Ingest data from a single API endpoint.
        
        Args:
            endpoint: API endpoint URL
            
        Returns:
            List of ingestion results
        """
        async with aiohttp.ClientSession() as session:
            data = await self._fetch_data(session, endpoint)
            
            if not data:
                return [{
                    'doc_id': None,
                    'content': None,
                    'metadata': None,
                    'status': 'failed',
                    'error': 'Failed to fetch data'
                }]
            
            results = []
            
            if isinstance(data, list):
                items = data
            elif isinstance(data, dict) and 'results' in data:
                items = data['results']
            elif isinstance(data, dict) and 'data' in data:
                items = data['data']
            else:
                items = [data]
            
            for item in items:
                content = self._serialize_item(item)
                content_hash = hashlib.md5(content.encode()).hexdigest()[:16]
                
                result = {
                    'doc_id': f"api_{content_hash}",
                    'content': content,
                    'metadata': {
                        'source_type': 'api',
                        'endpoint': endpoint,
                        'last_fetched': datetime.now().isoformat(),
                        'item_id': item.get('id', content_hash),
                        **self.metadata
                    },
                    'status': 'success'
                }
                
                results.append(result)
                await asyncio.sleep(self.rate_limit_delay)
            
            return results
    
    def _serialize_item(self, item: Dict[str, Any]) -> str:
        """Serialize an API item to text.
        
        Args:
            item: API response item
            
        Returns:
            Serialized text
        """
        def flatten_dict(d, parent_key='', sep=' '):
            items = []
            for k, v in d.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(flatten_dict(v, new_key, sep).items())
                elif isinstance(v, list):
                    items.append((new_key, ', '.join(map(str, v))))
                else:
                    items.append((new_key, str(v)))
            return dict(items)
        
        flat = flatten_dict(item)
        return '\n'.join(f"{k}: {v}" for k, v in flat.items())
    
    async def ingest_all(self) -> List[Dict[str, Any]]:
        """Ingest data from all configured endpoints.
        
        Returns:
            List of ingestion results
        """
        all_results = []
        
        for endpoint in self.endpoints:
            logger.info(f"Ingesting from API: {endpoint}")
            results = await self.ingest_endpoint(endpoint)
            all_results.extend(results)
            
        return all_results


class GraphQLConnector(APIConnector):
    """Connector for GraphQL API endpoints."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize GraphQL connector.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.graphql_query = config.get('graphql_query', '')
        self.variables = config.get('variables', {})
        
    async def _fetch_data(self, session: aiohttp.ClientSession, endpoint: str) -> Optional[Dict[str, Any]]:
        """Fetch data using GraphQL query.
        
        Args:
            session: aiohttp session
            endpoint: GraphQL endpoint URL
            
        Returns:
            GraphQL response or None if failed
        """
        try:
            payload = {
                'query': self.graphql_query,
                'variables': self.variables
            }
            
            async with session.post(
                endpoint,
                json=payload,
                headers=self.headers,
                timeout=self.timeout
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    if 'errors' in data:
                        logger.error(f"GraphQL errors: {data['errors']}")
                        return None
                    return data
                else:
                    logger.warning(f"Failed to fetch {endpoint}: HTTP {response.status}")
                    return None
        except Exception as e:
            logger.error(f"Error fetching {endpoint}: {e}")
            return None
