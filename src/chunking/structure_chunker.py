"""Structure-aware chunker that respects document structure."""

from typing import List, Dict, Any, Optional
import re
import logging

logger = logging.getLogger(__name__)


class StructureAwareChunker:
    """Chunker that splits text while respecting document structure."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize structure-aware chunker.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.chunk_size = config.get('chunk_size', 512)
        self.chunk_overlap = config.get('chunk_overlap', 50)
        self.max_sections = config.get('max_sections', 3)
        
    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Chunk text by respecting headings and sections.
        
        Args:
            text: Text to chunk
            metadata: Optional metadata to attach to chunks
            
        Returns:
            List of chunk dictionaries
        """
        if not text.strip():
            return []
        
        sections = self._split_into_sections(text)
        chunks = []
        
        for section in sections:
            section_chunks = self._chunk_section(section, metadata, len(chunks))
            chunks.extend(section_chunks)
        
        return chunks
    
    def _split_into_sections(self, text: str) -> List[str]:
        """Split text into sections based on headings.
        
        Args:
            text: Text to split
            
        Returns:
            List of section texts
        """
        sections = []
        current_section = ""
        
        lines = text.split('\n')
        i = 0
        
        while i < len(lines):
            line = lines[i].strip()
            
            if self._is_heading(line):
                if current_section:
                    sections.append(current_section.strip())
                current_section = line + '\n'
            else:
                current_section += lines[i] + '\n'
            
            i += 1
        
        if current_section:
            sections.append(current_section.strip())
        
        return sections
    
    def _is_heading(self, line: str) -> bool:
        """Check if a line is a heading.
        
        Args:
            line: Line to check
            
        Returns:
            True if line appears to be a heading
        """
        if not line:
            return False
        
        heading_patterns = [
            r'^#{1,6}\s+.+',  # Markdown headings
            r'^[A-Z][^.]*\n[=\-]+$',  # Underlined headings
            r'^\d+(\.\d+)*\s+[A-Z]',  # Numbered headings
            r'^[A-Z][^.]{0,50}$',  # Short capital lines
        ]
        
        for pattern in heading_patterns:
            if re.match(pattern, line):
                return True
        
        return False
    
    def _chunk_section(self, section: str, metadata: Optional[Dict[str, Any]], start_index: int) -> List[Dict[str, Any]]:
        """Chunk a single section.
        
        Args:
            section: Section text
            metadata: Metadata to attach
            start_index: Starting chunk index
            
        Returns:
            List of chunk dictionaries
        """
        if len(section) <= self.chunk_size:
            return [self._create_chunk(section, metadata, start_index)]
        
        chunks = []
        lines = section.split('\n')
        current_chunk = ""
        current_length = 0
        
        for line in lines:
            line_length = len(line) + 1
            
            if current_length + line_length <= self.chunk_size:
                if current_chunk:
                    current_chunk += '\n' + line
                else:
                    current_chunk = line
                current_length += line_length
            else:
                if current_chunk:
                    chunks.append(self._create_chunk(current_chunk, metadata, start_index + len(chunks)))
                
                if line_length > self.chunk_size:
                    chunks.extend(self._split_long_line(line, metadata, start_index + len(chunks)))
                    current_chunk = ""
                    current_length = 0
                else:
                    current_chunk = line
                    current_length = line_length
        
        if current_chunk:
            chunks.append(self._create_chunk(current_chunk, metadata, start_index + len(chunks)))
        
        return chunks
    
    def _split_long_line(self, line: str, metadata: Optional[Dict[str, Any]], start_index: int) -> List[Dict[str, Any]]:
        """Split a long line into multiple chunks.
        
        Args:
            line: Long line to split
            metadata: Metadata to attach
            start_index: Starting chunk index
            
        Returns:
            List of chunk dictionaries
        """
        chunks = []
        words = line.split()
        current_chunk = ""
        current_length = 0
        
        for word in words:
            word_length = len(word) + 1
            
            if current_length + word_length <= self.chunk_size:
                if current_chunk:
                    current_chunk += ' ' + word
                else:
                    current_chunk = word
                current_length += word_length
            else:
                if current_chunk:
                    chunks.append(self._create_chunk(current_chunk, metadata, start_index + len(chunks)))
                current_chunk = word
                current_length = word_length
        
        if current_chunk:
            chunks.append(self._create_chunk(current_chunk, metadata, start_index + len(chunks)))
        
        return chunks
    
    def _create_chunk(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        chunk_index: int = 0
    ) -> Dict[str, Any]:
        """Create a chunk dictionary.
        
        Args:
            text: Chunk text
            metadata: Original metadata
            chunk_index: Index of this chunk
            
        Returns:
            Chunk dictionary
        """
        return {
            'text': text,
            'chunk_index': chunk_index,
            'total_chunks': 0,
            'metadata': metadata or {}
        }
