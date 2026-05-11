FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and data
COPY . .

# Pre-build the TF-IDF index at build time for fast cold start
RUN python -c "from app.catalog import CatalogStore; from app.retriever import CatalogRetriever; store = CatalogStore(); retriever = CatalogRetriever(store); print(f'Catalog loaded: {len(store)} items, TF-IDF index built')"

# Clean up old FAISS/embedding artifacts if present
RUN rm -f data/faiss_index.bin data/embeddings.npy data/metadata.json

# Expose port
ENV PORT=8000
EXPOSE 8000

# Limit Python memory usage and disable unnecessary caching
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV MALLOC_TRIM_THRESHOLD_=65536

# Run with a single worker to minimize memory footprint
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--limit-max-requests", "1000"]
