"""
Enhanced Document Indexing for RAG System
Builds vector index with semantic chunking and overlap for better retrieval.
"""

import os
import glob
import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader
from typing import List, Tuple
import re


def semantic_chunk_text(text: str, chunk_size: int = 300, overlap: int = 30) -> List[str]:
    """
    Chunk text semantically with overlap for better context preservation.
    
    Args:
        text: Input text to chunk
        chunk_size: Target size for each chunk
        overlap: Number of characters to overlap between chunks
        
    Returns:
        List of text chunks
    """
    # Split into sentences (simple sentence tokenization)
    # This is a basic implementation; for production consider nltk or spacy
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence_length = len(sentence)
        
        # If adding this sentence exceeds chunk_size and we have content, finalize chunk
        if current_length + sentence_length > chunk_size and current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append(chunk_text)
            
            # Keep last few sentences for overlap
            overlap_sentences = []
            overlap_length = 0
            for sent in reversed(current_chunk):
                if overlap_length + len(sent) <= overlap:
                    overlap_sentences.insert(0, sent)
                    overlap_length += len(sent)
                else:
                    break
            
            current_chunk = overlap_sentences
            current_length = overlap_length
        
        current_chunk.append(sentence)
        current_length += sentence_length
    
    # Add remaining content
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return [c.strip() for c in chunks if c.strip()]


def load_documents(source_dir: str) -> Tuple[List[str], List[dict], List[str]]:
    """
    Load and chunk documents from source directory.
    
    Args:
        source_dir: Directory containing source documents
        
    Returns:
        Tuple of (documents, metadatas, ids)
    """
    documents = []
    metadatas = []
    ids = []
    
    files = glob.glob(os.path.join(source_dir, "*"))
    print(f"Found {len(files)} files in {source_dir}")
    
    for file_path in files:
        filename = os.path.basename(file_path)
        print(f"Processing {filename}...", end=" ")
        text = ""
        
        try:
            # PDF files
            if filename.lower().endswith(".pdf"):
                reader = PdfReader(file_path)
                for page in reader.pages:
                    text += page.extract_text() + "\n"
            
            # Text files (.txt, .md, etc.)
            elif filename.lower().endswith((".txt", ".md", ".text")):
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()
            
            # DOCX files (if python-docx is installed)
            elif filename.lower().endswith(".docx"):
                try:
                    from docx import Document
                    doc = Document(file_path)
                    text = "\n".join([para.text for para in doc.paragraphs])
                except ImportError:
                    print("python-docx not installed, skipping DOCX file")
                    continue
            
            else:
                print(f"Skipping unsupported file type")
                continue
            
            # Clean text
            text = text.strip()
            if not text:
                print("No text extracted, skipping")
                continue
            
            # Semantic chunking with overlap
            chunks = semantic_chunk_text(text, chunk_size=300, overlap=30)
            
            # Fallback to simple chunking if semantic chunking fails
            if not chunks:
                chunks = [text[i:i+300] for i in range(0, len(text), 270)]
            
            # Add chunks to collection
            for i, chunk in enumerate(chunks):
                if len(chunk) < 20:  # Skip very tiny chunks
                    continue
                    
                documents.append(chunk)
                metadatas.append({
                    "source": filename,
                    "chunk_id": i,
                    "total_chunks": len(chunks),
                    "char_count": len(chunk)
                })
                ids.append(f"{filename}_{i}")
            
            print(f"✓ {len(chunks)} chunks created")
                
        except Exception as e:
            print(f"✗ Error: {e}")
            
    return documents, metadatas, ids


def build_index():
    """Build RAG index from documents in raw_docs directory."""
    raw_docs_dir = "rag/raw_docs"
    persist_directory = "rag/chroma_db"
    
    if not os.path.exists(raw_docs_dir):
        print(f"Error: Directory {raw_docs_dir} does not exist.")
        print("Please create it and add your knowledge base documents.")
        return
    
    print("="*60)
    print("Building RAG Index for Mental Health Chatbot")
    print("="*60)
    print()

    print("Step 1: Loading and chunking documents...")
    documents, metadatas, ids = load_documents(raw_docs_dir)
    
    if not documents:
        print("\nNo documents loaded. Please add documents to rag/raw_docs/")
        return
    
    print(f"\nStep 2: Processed {len(documents)} total chunks")
    
    # Calculate statistics
    sources = set(m['source'] for m in metadatas)
    avg_chunk_size = sum(m['char_count'] for m in metadatas) / len(metadatas)
    
    print(f"  - {len(sources)} source documents")
    print(f"  - Average chunk size: {avg_chunk_size:.0f} characters")
    
    print("\nStep 3: Initializing ChromaDB...")
    client = chromadb.PersistentClient(path=persist_directory)
    
    # Use sentence-transformers for embeddings
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    
    # Delete existing collection if it exists (fresh start)
    try:
        client.delete_collection(name="mental_health_docs")
        print("  - Deleted existing collection")
    except:
        pass
    
    collection = client.create_collection(
        name="mental_health_docs",
        embedding_function=ef
    )
    print("  - Created new collection")
    
    print("\nStep 4: Adding documents to collection...")
    print("  This may take a few minutes...")
    
    # Add in batches to avoid memory issues and show progress
    batch_size = 50
    total_batches = (len(documents) + batch_size - 1) // batch_size
    
    for batch_num in range(total_batches):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, len(documents))
        
        collection.add(
            documents=documents[start_idx:end_idx],
            metadatas=metadatas[start_idx:end_idx],
            ids=ids[start_idx:end_idx]
        )
        
        progress = ((batch_num + 1) / total_batches) * 100
        print(f"  Progress: {progress:.0f}% ({end_idx}/{len(documents)} chunks)", end='\r')
    
    print(f"\n  ✓ All {len(documents)} chunks indexed successfully")
    
    # Verify
    final_count = collection.count()
    print(f"\nStep 5: Verification")
    print(f"  - Collection contains {final_count} chunks")
    print(f"  - Index saved to: {persist_directory}")
    
    print("\n" + "="*60)
    print("Index building complete!")
    print("="*60)
    print("\nYou can now use the RAG system in your chatbot.")
    print("Test it with: python rag/search.py")


if __name__ == "__main__":
    build_index()

