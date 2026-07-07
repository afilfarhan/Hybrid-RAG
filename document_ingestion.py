"""
Document Ingestion Utility

Supports PDF, Markdown, TXT, and other document formats.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import os

try:
    from unstructured.partition.pdf import partition_pdf
    from unstructured.partition.text import partition_text
    from unstructured.partition.md import partition_md
    from unstructured.chunking.title import chunk_by_title
    HAS_UNSTRUCTURED = True
except ImportError:
    HAS_UNSTRUCTURED = False


class DocumentIngestor:
    """Handles document ingestion and chunking."""
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 51,
        max_chunk_size: int = 1024
    ):
        """
        Initialize document ingester.
        
        Args:
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            max_chunk_size: Maximum chunk size
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_chunk_size = max_chunk_size
    
    def ingest_file(self, file_path: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Ingest a single file and return chunks.
        
        Args:
            file_path: Path to the file
            metadata: Optional metadata to attach to chunks
            
        Returns:
            List of chunks with text and metadata
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Get file extension
        ext = file_path.suffix.lower()
        
        # Read file content based on type
        if ext == '.pdf':
            if not HAS_UNSTRUCTURED:
                raise ImportError(
                    "unstructured not installed. "
                    "Install with: pip install unstructured[pdf] pdfminer.six pdfplumber"
                )
            try:
                elements = partition_pdf(file_path)
                text = "\n\n".join([str(el) for el in elements])
            except Exception as e:
                raise ImportError(
                    f"PDF processing failed. Install: pip install unstructured[pdf] pdfminer.six pdfplumber. Error: {e}"
                )
        elif ext == '.md' or ext == '.markdown':
            if not HAS_UNSTRUCTURED:
                # Fallback: read as text
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            else:
                elements = partition_md(file_path)
                text = "\n\n".join([str(el) for el in elements])
        elif ext == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        else:
            # Try to read as text
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        
        # Chunk the text
        chunks = self._chunk_text(text)
        
        # Add metadata to each chunk
        base_metadata = metadata or {}
        base_metadata["source"] = str(file_path)
        
        result = []
        for i, chunk_text in enumerate(chunks):
            chunk_metadata = base_metadata.copy()
            chunk_metadata["chunk_id"] = i
            chunk_metadata["total_chunks"] = len(chunks)
            
            result.append({
                "text": chunk_text,
                "metadata": chunk_metadata
            })
        
        return result
    
    def ingest_directory(self, directory: str, recursive: bool = True) -> List[Dict[str, Any]]:
        """
        Ingest all documents in a directory.
        
        Args:
            directory: Path to directory
            recursive: Whether to search subdirectories
            
        Returns:
            List of all chunks from all documents
        """
        directory = Path(directory)
        all_chunks = []
        
        # Supported extensions
        extensions = {'.pdf', '.md', '.markdown', '.txt'}
        
        # Find all files
        if recursive:
            files = directory.rglob('*')
        else:
            files = directory.glob('*')
        
        for file_path in files:
            if file_path.is_file() and file_path.suffix.lower() in extensions:
                # Skip README files
                if file_path.name.lower().startswith("readme"):
                    continue
                
                try:
                    chunks = self.ingest_file(file_path)
                    all_chunks.extend(chunks)
                    print(f"[OK] Ingested: {file_path} ({len(chunks)} chunks)")
                except ImportError as e:
                    print(f"[SKIP] PDF support not available for {file_path.name}: {e}")
                    print(f"       Install: pip install unstructured[pdf] pdfminer.six pdfplumber")
                except Exception as e:
                    print(f"[ERROR] Error ingesting {file_path}: {e}")
        
        return all_chunks
    
    def _chunk_text(self, text: str) -> List[str]:
        """
        Chunk text into smaller pieces.
        
        Args:
            text: Text to chunk
            
        Returns:
            List of text chunks
        """
        # Simple chunking by character count
        chunks = []
        words = text.split()
        current_chunk = []
        current_length = 0
        
        for word in words:
            word_length = len(word) + 1  # +1 for space
            
            if current_length + word_length > self.chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_length = word_length
            else:
                current_chunk.append(word)
                current_length += word_length
        
        # Add remaining chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks


def create_ingestion_script():
    """Create a sample ingestion script."""
    script = '''
"""
Sample script to ingest PDF files into ChromaDB.
"""

from services.chromadb_store import ChromaDBVectorStore
from services.embedding_inmemory import InMemoryEmbeddingService
from services.retriever_inmemory import InMemoryRetrievalService
from document_ingestion import DocumentIngestor, HAS_UNSTRUCTURED

def main():
    print("=" * 60)
    print("Document Ingestion to ChromaDB")
    print("=" * 60)
    
    # Check dependencies
    if not HAS_UNSTRUCTURED:
        print("\\n⚠️  WARNING: unstructured not installed")
        print("Install with: pip install unstructured[pdf]")
        print("This script will still work but PDF support may be limited\\n")
    
    # Initialize vector store
    print("Initializing ChromaDB vector store...")
    vector_store = ChromaDBVectorStore(
        persist_path="./data/vector_store",
        collection_name="documents",
        dimension=384
    )
    print(f"[OK] Collection created: {vector_store.collection_name}")
    print(f"[OK] Current count: {vector_store.get_stats()['count']} documents")
    
    # Initialize ingester
    print("\\nInitializing document ingester...")
    ingester = DocumentIngestor(
        chunk_size=512,
        chunk_overlap=51
    )
    print("[OK] Document ingester ready")
    
    # Ingest files
    print("\\n" + "=" * 60)
    print("Ingesting Documents")
    print("=" * 60)
    
    # Option 1: Ingest a single file
    print("\\n1. Ingesting single file...")
    single_file_chunks = ingester.ingest_file(
        "./data/sample_docs/returns_policy.md",
        metadata={"doc_type": "policy"}
    )
    print(f"   Created {len(single_file_chunks)} chunks")
    
    # Add to vector store
    texts = [c["text"] for c in single_file_chunks]
    metadatas = [c["metadata"] for c in single_file_chunks]
    vector_store.add_batch(texts, metadatas)
    print(f"   Added to vector store: {len(single_file_chunks)} chunks")
    
    # Option 2: Ingest entire directory
    print("\\n2. Ingesting directory...")
    all_chunks = ingester.ingest_directory("./data/sample_docs", recursive=True)
    print(f"   Total chunks created: {len(all_chunks)}")
    
    # Add to vector store
    if all_chunks:
        texts = [c["text"] for c in all_chunks]
        metadatas = [c["metadata"] for c in all_chunks]
        vector_store.add_batch(texts, metadatas)
        print(f"   Added to vector store: {len(all_chunks)} chunks")
    
    # Final stats
    print("\\n" + "=" * 60)
    print("Final Statistics")
    print("=" * 60)
    stats = vector_store.get_stats()
    print(f"Collection: {stats['name']}")
    print(f"Document count: {stats['count']}")
    print(f"Dimension: {stats['dimension']}")
    
    # Persist
    print("\\nPersisting database...")
    vector_store.persist()
    print("[OK] Database persisted to disk")
    
    print("\\n" + "=" * 60)
    print("Ingestion Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
'''
    
    return script


if __name__ == "__main__":
    print(create_ingestion_script())
