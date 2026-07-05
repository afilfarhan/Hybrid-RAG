"""Text preprocessor for cleaning and normalizing text."""

import re
import html
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class TextPreprocessor:
    """Preprocessor for cleaning and normalizing text."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize preprocessor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.remove_html = config.get('remove_html', True)
        self.remove_special_chars = config.get('remove_special_chars', False)
        self.normalize_whitespace = config.get('normalize_whitespace', True)
        self.lowercase = config.get('lowercase', False)
        self.remove_pii = config.get('remove_pii', True)
        
    def preprocess(self, text: str) -> str:
        """Preprocess text.
        
        Args:
            text: Text to preprocess
            
        Returns:
            Preprocessed text
        """
        if not text:
            return ""
        
        if self.remove_html:
            text = self._remove_html(text)
        
        if self.normalize_whitespace:
            text = self._normalize_whitespace(text)
        
        if self.lowercase:
            text = text.lower()
        
        if self.remove_special_chars:
            text = self._remove_special_chars(text)
        
        if self.remove_pii:
            text = self._remove_pii(text)
        
        return text.strip()
    
    def _remove_html(self, text: str) -> str:
        """Remove HTML tags from text.
        
        Args:
            text: Text with HTML
            
        Returns:
            Text without HTML tags
        """
        import re
        
        clean = re.compile('<.*?>')
        text = re.sub(clean, '', text)
        text = html.unescape(text)
        
        return text
    
    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace in text.
        
        Args:
            text: Text to normalize
            
        Returns:
            Text with normalized whitespace
        """
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n+', '\n', text)
        return text.strip()
    
    def _remove_special_chars(self, text: str) -> str:
        """Remove special characters from text.
        
        Args:
            text: Text to clean
            
        Returns:
            Text without special characters
        """
        text = re.sub(r'[^\w\s\.]', '', text)
        return text
    
    def _remove_pii(self, text: str) -> str:
        """Remove personally identifiable information.
        
        Args:
            text: Text to clean
            
        Returns:
            Text with PII redacted
        """
        patterns = [
            (r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]'),
            (r'\b\d{10}\b', '[PHONE]'),
            (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]'),
            (r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b', '[CARD]'),
        ]
        
        for pattern, replacement in patterns:
            text = re.sub(pattern, replacement, text)
        
        return text
    
    def preprocess_batch(self, texts: List[str]) -> List[str]:
        """Preprocess a batch of texts.
        
        Args:
            texts: List of texts to preprocess
            
        Returns:
            List of preprocessed texts
        """
        return [self.preprocess(text) for text in texts]
