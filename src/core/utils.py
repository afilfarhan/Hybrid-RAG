"""
Hybrid RAG - Core utilities
"""

import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def generate_id(text: str, prefix: str = "") -> str:
    """Generate a deterministic ID from text."""
    hash_val = hashlib.md5(text.encode()).hexdigest()
    return f"{prefix}{hash_val}" if prefix else hash_val


def clean_text(text: str) -> str:
    """Clean and normalize text."""
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text)
    # Remove leading/trailing whitespace
    text = text.strip()
    return text


def format_timestamp() -> str:
    """Get current timestamp in ISO format."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def safe_json_loads(text: str) -> Optional[Dict[str, Any]]:
    """Safely parse JSON string."""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None


def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """Split a list into chunks of specified size."""
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def merge_dicts(*dicts: Dict[str, Any]) -> Dict[str, Any]:
    """Merge multiple dictionaries, with later ones taking precedence."""
    result = {}
    for d in dicts:
        result.update(d)
    return result
