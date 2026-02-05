"""
Enhanced RAG System for Mental Health Chatbot
Provides context retrieval with query expansion, reranking, and caching.
"""

import chromadb
from chromadb.utils import embedding_functions
import os
import logging
from typing import List, Dict, Optional
from functools import lru_cache
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGSystem:
    def __init__(self, persist_directory="rag/chroma_db", use_reranker=False):
        """
        Initialize RAG system.
        
        Args:
            persist_directory: Path to ChromaDB directory
            use_reranker: Whether to use cross-encoder reranking (slower but more accurate)
        """
        self.persist_directory = persist_directory
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self.collection = self.client.get_or_create_collection(
            name="mental_health_docs",
            embedding_function=self.ef
        )
        
        # Optional reranker for better accuracy
        self.use_reranker = use_reranker
        self.reranker = None
        if use_reranker:
            try:
                from sentence_transformers import CrossEncoder
                self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
                logger.info("Reranker loaded successfully")
            except ImportError:
                logger.warning("sentence-transformers not found. Reranking disabled.")
                self.use_reranker = False
        
        # Simple query cache
        self.cache = {}
        self.cache_max_size = 100
        
        logger.info(f"RAG System initialized. DB: {persist_directory}, Reranker: {use_reranker}")

    def expand_query(self, query: str) -> List[str]:
        """
        Generate alternative phrasings of the query for better retrieval.
        
        Args:
            query: Original user query
            
        Returns:
            List of query variations (including original)
        """
        queries = [query]
        
        # Simple synonym expansions for common mental health terms
        expansions = {
            "anxious": ["worried", "nervous", "stressed"],
            "depressed": ["sad", "down", "low mood"],
            "panic": ["anxiety attack", "overwhelming fear"],
            "worried": ["anxious", "concerned", "stressed"],
            "sad": ["depressed", "unhappy", "down"],
        }
        
        query_lower = query.lower()
        for term, synonyms in expansions.items():
            if term in query_lower:
                for synonym in synonyms[:1]:  # Add just one synonym to avoid too many queries
                    expanded = query_lower.replace(term, synonym)
                    if expanded != query_lower:
                        queries.append(expanded)
                break  # Only expand first matched term
        
        return queries[:3]  # Limit to 3 queries max

    def get_context(self, query: str, k: int = 3, filter_metadata: Dict = None,
                   use_query_expansion: bool = True) -> str:
        """
        Retrieve relevant context from knowledge base.
        
        Args:
            query: User query
            k: Number of results to return
            filter_metadata: Optional metadata filter (e.g., {"source": "anxiety_guide.pdf"})
            use_query_expansion: Whether to expand query with synonyms
            
        Returns:
            Formatted context string
        """
        # Check cache first
        cache_key = self._get_cache_key(query, k, filter_metadata)
        if cache_key in self.cache:
            logger.debug(f"Cache hit for query: {query[:50]}...")
            return self.cache[cache_key]
        
        # Generate query variations
        queries = self.expand_query(query) if use_query_expansion else [query]
        
        # Retrieve documents for all query variations
        all_docs = []
        all_metadatas = []
        all_distances = []
        
        for q in queries:
            try:
                results = self.collection.query(
                    query_texts=[q],
                    n_results=k * 2,  # Get more candidates for reranking
                    where=filter_metadata
                )
                
                if results['documents'] and results['documents'][0]:
                    all_docs.extend(results['documents'][0])
                    all_metadatas.extend(results['metadatas'][0])
                    all_distances.extend(results['distances'][0])
            except Exception as e:
                logger.error(f"Error querying with '{q}': {e}")
        
        if not all_docs:
            logger.warning(f"No documents found for query: {query}")
            return ""
        
        # Remove duplicates while preserving order
        unique_docs = []
        unique_metas = []
        unique_dists = []
        seen = set()
        
        for doc, meta, dist in zip(all_docs, all_metadatas, all_distances):
            doc_hash = hashlib.md5(doc.encode()).hexdigest()
            if doc_hash not in seen:
                seen.add(doc_hash)
                unique_docs.append(doc)
                unique_metas.append(meta)
                unique_dists.append(dist)
        
        # Rerank if enabled
        if self.use_reranker and self.reranker and len(unique_docs) > 1:
            ranked_docs, ranked_metas = self._rerank(query, unique_docs, unique_metas)
        else:
            # Sort by distance (lower is better)
            sorted_items = sorted(
                zip(unique_dists, unique_docs, unique_metas),
                key=lambda x: x[0]
            )
            ranked_docs = [item[1] for item in sorted_items[:k]]
            ranked_metas = [item[2] for item in sorted_items[:k]]
        
        # Format context
        context = self._format_context(ranked_docs[:k], ranked_metas[:k])
        
        # Cache result
        if len(self.cache) >= self.cache_max_size:
            # Simple FIFO cache eviction
            self.cache.pop(next(iter(self.cache)))
        self.cache[cache_key] = context
        
        return context

    def _rerank(self, query: str, documents: List[str], metadatas: List[Dict]) -> tuple:
        """Rerank documents using cross-encoder."""
        pairs = [[query, doc] for doc in documents]
        scores = self.reranker.predict(pairs)
        
        # Sort by score (descending)
        ranked = sorted(
            zip(scores, documents, metadatas),
            key=lambda x: x[0],
            reverse=True
        )
        
        return [item[1] for item in ranked], [item[2] for item in ranked]

    def _format_context(self, documents: List[str], metadatas: List[Dict]) -> str:
        """Format retrieved documents into context string."""
        # Simple join with minimal separators to save tokens
        return "\n\n".join([doc.strip() for doc in documents])

    def _get_cache_key(self, query: str, k: int, filter_metadata: Optional[Dict]) -> str:
        """Generate cache key from query parameters."""
        key_parts = [query, str(k)]
        if filter_metadata:
            key_parts.append(str(sorted(filter_metadata.items())))
        return hashlib.md5("".join(key_parts).encode()).hexdigest()

    def get_stats(self) -> Dict:
        """Get RAG system statistics."""
        try:
            count = self.collection.count()
            return {
                "total_chunks": count,
                "cache_size": len(self.cache),
                "reranker_enabled": self.use_reranker
            }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {}


# Singleton instance for easy import
_rag_instance = None

def get_rag_context(query: str, k: int = 2, use_reranker: bool = False) -> str:
    """
    Convenience function for getting RAG context.
    
    Args:
        query: User query
        k: Number of context chunks to retrieve
        use_reranker: Whether to use reranking (slower but better)
        
    Returns:
        Formatted context string
    """
    global _rag_instance
    if _rag_instance is None:
        # Check if DB exists
        if not os.path.exists("rag/chroma_db"):
            logger.warning("RAG database not found. Run python rag/build_index.py first.")
            return ""
        _rag_instance = RAGSystem(use_reranker=use_reranker)
    return _rag_instance.get_context(query, k=k)


def get_rag_stats() -> Dict:
    """Get RAG system statistics."""
    global _rag_instance
    if _rag_instance is None:
        return {"status": "not_initialized"}
    return _rag_instance.get_stats()
