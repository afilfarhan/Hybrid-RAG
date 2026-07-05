"""File-based data connector for ingestion pipeline."""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
import hashlib
import json
from abc import ABC

logger = logging.getLogger(__name__)


class FileConnector(ABC):
    """Base connector for file-based data sources."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize file connector.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.source_dir = Path(config.get('source_dir', './data/sources'))
        self.supported_extensions = config.get('supported_extensions', ['.pdf', '.docx', '.txt', '.md', '.html'])
        self.metadata = config.get('metadata', {})
        
    def _generate_document_id(self, file_path: Path, content: str) -> str:
        """Generate unique document ID.
        
        Args:
            file_path: Path to the file
            content: Document content
            
        Returns:
            Unique document ID
        """
        content_hash = hashlib.md5(content.encode()).hexdigest()[:16]
        file_hash = hashlib.md5(str(file_path).encode()).hexdigest()[:8]
        return f"doc_{file_hash}_{content_hash}"
    
    def _extract_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Extract metadata from file path.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Metadata dictionary
        """
        return {
            'source_type': 'file',
            'file_path': str(file_path),
            'file_name': file_path.name,
            'file_extension': file_path.suffix,
            'file_size': file_path.stat().st_size,
            'last_modified': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
            **self.metadata
        }
    
    def _get_files(self, directory: Optional[Path] = None) -> List[Path]:
        """Get all supported files in directory.
        
        Args:
            directory: Directory to search (defaults to source_dir)
            
        Returns:
            List of file paths
        """
        dir_path = directory or self.source_dir
        files = []
        
        if not dir_path.exists():
            logger.warning(f"Directory does not exist: {dir_path}")
            return files
            
        for ext in self.supported_extensions:
            files.extend(dir_path.rglob(f"*{ext}"))
            
        return files
    
    async def ingest_file(self, file_path: Path) -> Dict[str, Any]:
        """Ingest a single file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Ingestion result
        """
        try:
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            content = await self._read_file(file_path)
            metadata = self._extract_metadata(file_path)
            doc_id = self._generate_document_id(file_path, content)
            
            return {
                'doc_id': doc_id,
                'content': content,
                'metadata': metadata,
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"Error ingesting file {file_path}: {e}")
            return {
                'doc_id': None,
                'content': None,
                'metadata': None,
                'status': 'failed',
                'error': str(e)
            }
    
    async def ingest_directory(self, directory: Optional[Path] = None) -> List[Dict[str, Any]]:
        """Ingest all files in directory.
        
        Args:
            directory: Directory to ingest (defaults to source_dir)
            
        Returns:
            List of ingestion results
        """
        files = self._get_files(directory)
        results = []
        
        for file_path in files:
            result = await self.ingest_file(file_path)
            results.append(result)
            
            if result['status'] == 'success':
                logger.info(f"Ingested: {file_path}")
            else:
                logger.error(f"Failed to ingest: {file_path} - {result.get('error', 'Unknown error')}")
        
        return results
    
    @abstractmethod
    async def _read_file(self, file_path: Path) -> str:
        """Read file content (to be implemented by subclasses).
        
        Args:
            file_path: Path to the file
            
        Returns:
            File content as string
        """
        pass


class PDFConnector(FileConnector):
    """PDF file connector."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize PDF connector.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.supported_extensions = ['.pdf']
        
    async def _read_file(self, file_path: Path) -> str:
        """Read PDF file content using pdfplumber.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            PDF content as string
        """
        import pdfplumber
        
        content_parts = []
        
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                page_text = page.extract_text() or ""
                content_parts.append(f"[Page {page_num}]\n{page_text}")
                
        return "\n\n".join(content_parts)


class DocumentConnector(FileConnector):
    """Word document connector (.docx)."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Word document connector.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.supported_extensions = ['.docx']
        
    async def _read_file(self, file_path: Path) -> str:
        """Read Word document content.
        
        Args:
            file_path: Path to the Word document
            
        Returns:
            Document content as string
        """
        from docx import Document
        
        doc = Document(file_path)
        content_parts = []
        
        for para in doc.paragraphs:
            if para.text.strip():
                content_parts.append(para.text)
                
        return "\n\n".join(content_parts)


class TextConnector(FileConnector):
    """Text file connector (.txt, .md)."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize text file connector.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.supported_extensions = ['.txt', '.md']
        
    async def _read_file(self, file_path: Path) -> str:
        """Read text file content.
        
        Args:
            file_path: Path to the text file
            
        Returns:
            File content as string
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()


class HTMLConnector(FileConnector):
    """HTML file connector."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize HTML file connector.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.supported_extensions = ['.html', '.htm']
        
    async def _read_file(self, file_path: Path) -> str:
        """Read HTML file content and extract text.
        
        Args:
            file_path: Path to the HTML file
            
        Returns:
            Extracted text content
        """
        from bs4 import BeautifulSoup
        
        with open(file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
            
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(['script', 'style']):
            script.decompose()
            
        text = soup.get_text()
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        
        return "\n".join(lines)
