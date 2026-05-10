"""
FAISS-based semantic retriever for the SHL product catalog.

Uses sentence-transformers for embeddings and FAISS for fast similarity search.
Implements hybrid search: semantic similarity + keyword boosting.
"""

import os
import numpy as np
import json
from typing import Optional

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
INDEX_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
INDEX_PATH = os.path.join(INDEX_DIR, "faiss_index.bin")
EMBEDDINGS_PATH = os.path.join(INDEX_DIR, "embeddings.npy")
METADATA_PATH = os.path.join(INDEX_DIR, "metadata.json")


class CatalogRetriever:
    """Hybrid semantic + keyword retriever over the SHL catalog."""
    
    def __init__(self, catalog_store):
        """Initialize retriever with a CatalogStore instance.
        
        Args:
            catalog_store: CatalogStore with loaded catalog items.
        """
        self.catalog = catalog_store
        self.model = None
        self.index = None
        self.embeddings = None
        self._build_or_load_index()
    
    def _get_model(self):
        """Lazy-load the sentence transformer model."""
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        return self.model
    
    def _build_or_load_index(self):
        """Build FAISS index from catalog or load from disk."""
        import faiss
        
        if os.path.exists(INDEX_PATH) and os.path.exists(EMBEDDINGS_PATH):
            # Load pre-built index
            self.index = faiss.read_index(INDEX_PATH)
            self.embeddings = np.load(EMBEDDINGS_PATH)
            return
        
        # Build from scratch
        model = self._get_model()
        
        # Generate embedding texts
        texts = [self.catalog.get_embedding_text(item) for item in self.catalog.items]
        
        # Encode all items
        self.embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
        self.embeddings = self.embeddings.astype(np.float32)
        
        # Build FAISS index (inner product on normalized vectors = cosine similarity)
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.embeddings)
        
        # Save to disk
        os.makedirs(INDEX_DIR, exist_ok=True)
        faiss.write_index(self.index, INDEX_PATH)
        np.save(EMBEDDINGS_PATH, self.embeddings)
    
    def retrieve(self, query: str, top_k: int = 20) -> list[dict]:
        """Retrieve top-k catalog items matching the query.
        
        Uses hybrid search: semantic similarity from FAISS + keyword boosting
        for exact name matches.
        
        Args:
            query: The search query (can be multi-sentence context summary).
            top_k: Number of results to return.
            
        Returns:
            List of catalog items sorted by relevance score, each with an added
            'score' field.
        """
        model = self._get_model()
        
        # Encode query
        query_embedding = model.encode([query], normalize_embeddings=True).astype(np.float32)
        
        # Search FAISS (get more than top_k to allow for re-ranking)
        fetch_k = min(top_k * 3, len(self.catalog.items))
        scores, indices = self.index.search(query_embedding, fetch_k)
        
        # Build results with hybrid scoring
        results = []
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            item = self.catalog.items[idx].copy()
            
            # Keyword boost: if query contains words that match item name
            name_lower = item["name"].lower()
            name_words = set(name_lower.replace("(", "").replace(")", "").replace("-", " ").split())
            
            # Boost for exact technology/skill name matches
            overlap = query_words & name_words
            # Filter out common words
            meaningful_overlap = overlap - {"new", "the", "a", "an", "and", "or", "for", "in", "of", "to", "with", "is", "are", "that", "this"}
            
            keyword_boost = len(meaningful_overlap) * 0.08
            
            # Extra boost if the full product name appears in query
            if name_lower in query_lower:
                keyword_boost += 0.2
            
            item["score"] = float(score) + keyword_boost
            results.append(item)
        
        # Sort by score descending and return top_k
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]
    
    def retrieve_by_names(self, names: list[str]) -> list[dict]:
        """Retrieve specific catalog items by their names.
        
        Used when the agent needs to compare specific assessments.
        Falls back to fuzzy matching if exact match fails.
        
        Args:
            names: List of assessment names to look up.
            
        Returns:
            List of matching catalog items.
        """
        results = []
        for name in names:
            item = self.catalog.get_by_name(name)
            if item:
                results.append(item)
            else:
                # Try substring match
                matches = self.catalog.search_by_name(name)
                if matches:
                    results.append(matches[0])
        return results
