"""
Lightweight TF-IDF retriever for the SHL product catalog.

Uses scikit-learn's TfidfVectorizer + cosine similarity instead of
sentence-transformers + FAISS.  This cuts ~400 MB of RAM (PyTorch +
transformer model) so the app fits comfortably on Render's free tier.
"""

import os
import re
import pickle
import numpy as np
from typing import Optional

INDEX_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
INDEX_PATH = os.path.join(INDEX_DIR, "tfidf_index.pkl")


class CatalogRetriever:
    """Hybrid TF-IDF + keyword retriever over the SHL catalog."""

    def __init__(self, catalog_store):
        """Initialize retriever with a CatalogStore instance.

        Args:
            catalog_store: CatalogStore with loaded catalog items.
        """
        self.catalog = catalog_store
        self.vectorizer = None
        self.tfidf_matrix = None
        self._build_or_load_index()

    def _build_or_load_index(self):
        """Build TF-IDF index from catalog or load from disk."""
        if os.path.exists(INDEX_PATH):
            try:
                with open(INDEX_PATH, "rb") as f:
                    data = pickle.load(f)
                self.vectorizer = data["vectorizer"]
                self.tfidf_matrix = data["matrix"]
                # Verify the index matches the current catalog size
                if self.tfidf_matrix.shape[0] == len(self.catalog.items):
                    return
            except Exception:
                pass  # Rebuild on any load error

        # Build from scratch
        from sklearn.feature_extraction.text import TfidfVectorizer

        texts = [self.catalog.get_embedding_text(item) for item in self.catalog.items]

        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words="english",
            sublinear_tf=True,
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)

        # Save to disk
        os.makedirs(INDEX_DIR, exist_ok=True)
        with open(INDEX_PATH, "wb") as f:
            pickle.dump({"vectorizer": self.vectorizer, "matrix": self.tfidf_matrix}, f)

    def retrieve(self, query: str, top_k: int = 20) -> list[dict]:
        """Retrieve top-k catalog items matching the query.

        Uses hybrid search: TF-IDF cosine similarity + keyword boosting
        for exact name matches.

        Args:
            query: The search query (can be multi-sentence context summary).
            top_k: Number of results to return.

        Returns:
            List of catalog items sorted by relevance score, each with an added
            'score' field.
        """
        # Transform query
        query_vec = self.vectorizer.transform([query])

        # Compute cosine similarities (tfidf_matrix rows are already L2-normed by TfidfVectorizer)
        scores = (self.tfidf_matrix @ query_vec.T).toarray().flatten()

        # Get top candidates (fetch more than needed for re-ranking)
        fetch_k = min(top_k * 3, len(self.catalog.items))
        top_indices = np.argsort(scores)[::-1][:fetch_k]

        # Build results with hybrid scoring
        results = []
        query_lower = query.lower()
        query_words = set(query_lower.split())

        for idx in top_indices:
            score = scores[idx]
            if score <= 0:
                continue

            item = self.catalog.items[idx].copy()

            # Keyword boost: if query contains words that match item name
            name_lower = item["name"].lower()
            name_words = set(name_lower.replace("(", "").replace(")", "").replace("-", " ").split())

            # Boost for exact technology/skill name matches
            overlap = query_words & name_words
            # Filter out common words
            meaningful_overlap = overlap - {
                "new", "the", "a", "an", "and", "or", "for", "in",
                "of", "to", "with", "is", "are", "that", "this",
            }

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
