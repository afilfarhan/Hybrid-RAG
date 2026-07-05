"""Ingestion scheduler for automated document ingestion."""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from pathlib import Path
import schedule
import threading
import time

logger = logging.getLogger(__name__)


class IngestionScheduler:
    """Scheduler for automated ingestion tasks."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize ingestion scheduler.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.running = False
        self.scheduler = schedule.JobQueue()
        
    def add_source_job(
        self,
        source_id: str,
        source_config: Dict[str, Any],
        schedule_interval: str,
        source_type: str = 'file'
    ) -> bool:
        """Add a scheduled ingestion job.
        
        Args:
            source_id: Unique source identifier
            source_config: Source configuration
            schedule_interval: Schedule interval (e.g., 'daily', 'hourly', 'every_30_minutes')
            source_type: Type of source (file, web, api)
            
        Returns:
            True if job was added successfully
        """
        self.jobs[source_id] = {
            'config': source_config,
            'interval': schedule_interval,
            'type': source_type,
            'last_run': None,
            'next_run': None,
            'status': 'scheduled'
        }
        
        logger.info(f"Added scheduled job for source '{source_id}' with interval: {schedule_interval}")
        return True
    
    def remove_job(self, source_id: str) -> bool:
        """Remove a scheduled job.
        
        Args:
            source_id: Source identifier
            
        Returns:
            True if job was removed successfully
        """
        if source_id in self.jobs:
            del self.jobs[source_id]
            logger.info(f"Removed scheduled job for source '{source_id}'")
            return True
        return False
    
    def get_jobs(self) -> List[Dict[str, Any]]:
        """Get list of all scheduled jobs.
        
        Returns:
            List of job configurations
        """
        return [
            {
                'source_id': source_id,
                **job_config
            }
            for source_id, job_config in self.jobs.items()
        ]
    
    async def run_job(self, source_id: str) -> Dict[str, Any]:
        """Run a single ingestion job.
        
        Args:
            source_id: Source identifier
            
        Returns:
            Ingestion results
        """
        if source_id not in self.jobs:
            raise ValueError(f"Unknown source: {source_id}")
        
        job = self.jobs[source_id]
        job['status'] = 'running'
        job['last_run'] = datetime.now().isoformat()
        
        try:
            from .file_connector import PDFConnector, DocumentConnector, TextConnector
            from .web_connector import WebConnector
            from .api_connector import APIConnector
            
            source_type = job['type']
            config = job['config']
            
            if source_type == 'file':
                connector = self._create_file_connector(config)
            elif source_type == 'web':
                connector = WebConnector(config)
            elif source_type == 'api':
                connector = APIConnector(config)
            else:
                raise ValueError(f"Unknown source type: {source_type}")
            
            results = await connector.ingest_all()
            
            success_count = sum(1 for r in results if r.get('status') == 'success')
            failed_count = len(results) - success_count
            
            job['status'] = 'completed'
            job['last_run'] = datetime.now().isoformat()
            
            return {
                'source_id': source_id,
                'status': 'completed',
                'results': results,
                'summary': {
                    'total': len(results),
                    'success': success_count,
                    'failed': failed_count
                }
            }
            
        except Exception as e:
            logger.error(f"Error running job for source '{source_id}': {e}")
            job['status'] = 'failed'
            job['last_error'] = str(e)
            return {
                'source_id': source_id,
                'status': 'failed',
                'error': str(e)
            }
    
    def _create_file_connector(self, config: Dict[str, Any]) -> Any:
        """Create appropriate file connector based on config.
        
        Args:
            config: Source configuration
            
        Returns:
            File connector instance
        """
        extensions = config.get('extensions', [])
        
        if '.pdf' in extensions:
            return PDFConnector(config)
        elif '.docx' in extensions:
            return DocumentConnector(config)
        else:
            return TextConnector(config)
    
    def start(self):
        """Start the scheduler."""
        self.running = True
        logger.info("Ingestion scheduler started")
        
        def run_scheduler():
            while self.running:
                schedule.run_pending()
                time.sleep(60)
        
        self.scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
        self.scheduler_thread.start()
    
    def stop(self):
        """Stop the scheduler."""
        self.running = False
        if hasattr(self, 'scheduler_thread'):
            self.scheduler_thread.join(timeout=5)
        logger.info("Ingestion scheduler stopped")
    
    async def run_all(self) -> List[Dict[str, Any]]:
        """Run all scheduled jobs.
        
        Returns:
            List of ingestion results
        """
        results = []
        
        for source_id in self.jobs:
            result = await self.run_job(source_id)
            results.append(result)
            
        return results
