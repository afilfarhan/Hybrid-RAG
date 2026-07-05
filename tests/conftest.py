"""Test fixtures for Hybrid RAG system."""

import pytest
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))


@pytest.fixture
def sample_text():
    """Sample text for testing."""
    return """# Welcome to Our Store

## Products
We offer a wide range of products.

## Return Policy
You can return items within 30 days.

## Contact
Email: support@example.com
Phone: 123-456-7890
"""


@pytest.fixture
def sample_faq():
    """Sample FAQ for testing."""
    return """# FAQ

## Order Status
**Q: How do I check my order?**
A: Log into your account and view order history.

## Returns
**Q: What is your return policy?**
A: 30 days from purchase.

## Shipping
**Q: How long does shipping take?**
A: 3-5 business days.
"""


@pytest.fixture
def sample_product():
    """Sample product data for testing."""
    return """# Product Catalog

## Product: Wireless Headphones
ID: WH-001
Price: $299.99
Description: High-quality wireless headphones.

## Product: Smart Watch
ID: SW-002
Price: $399.99
Description: Advanced smartwatch with health features.
"""


@pytest.fixture
def sample_html():
    """Sample HTML for testing."""
    return """<!DOCTYPE html>
<html>
<head><title>Test Page</title></head>
<body>
<h1>Welcome</h1>
<p>This is a test page.</p>
</body>
</html>
"""
