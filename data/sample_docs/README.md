# Hybrid RAG - Sample Documents

This directory contains sample documents for testing the RAG system.

## Sample Documents

### 1. product_catalog.md
Product catalog with product IDs, prices, descriptions, and features.

### 2. returns_policy.md
Return policy document with 30-day return window.

### 3. shipping_info.md
Shipping information including delivery times and costs.

### 4. faq.md
Frequently asked questions with Q&A format.

## Usage

Run the sample document creation script:

```bash
python create_sample_docs.py
```

This will create the sample documents in `data/sample_docs/`.

## Testing

Use the sample documents to test the ingestion pipeline:

```python
from src.ingestion.file_connector import TextConnector

connector = TextConnector({
    'source_dir': './data/sample_docs',
    'supported_extensions': ['.md']
})

files = connector._get_files()
print(f"Found {len(files)} files")
```
