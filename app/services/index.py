# app/services/index.py
import os
import numpy as np
import faiss
from app.config import settings
from app.services.storage import download_blob_to_file # <-- NEW IMPORT

def _idx(event: str) -> str:
    return os.path.join(settings.MEDIA_ROOT, "indices", f"{event}.faiss")

def _ids(event: str) -> str:
    return os.path.join(settings.MEDIA_ROOT, "indices", f"{event}.ids.npy")

def load_or_create_index(dim: int = 512, metric: str = "", event_slug: str = ""):
    metric = metric or settings.FAISS_METRIC
    event_slug = event_slug or settings.EVENT_SLUG
    index_path, ids_path = _idx(event_slug), _ids(event_slug)

    # 1. Check if files exist locally (as a cache)
    if os.path.exists(index_path) and os.path.exists(ids_path):
        print(f"Loading index for '{event_slug}' from local cache.")
        return faiss.read_index(index_path), np.load(ids_path)

    # 2. If not, try to download from Azure
    print(f"Local index for '{event_slug}' not found. Attempting to download from Azure...")
    index_downloaded = download_blob_to_file(
        public_id=os.path.basename(index_path),
        local_path=index_path,
        container_name=settings.AZURE_INDEX_CONTAINER
    )
    ids_downloaded = download_blob_to_file(
        public_id=os.path.basename(ids_path),
        local_path=ids_path,
        container_name=settings.AZURE_INDEX_CONTAINER
    )
    
    if index_downloaded and ids_downloaded:
        print("Successfully downloaded index from Azure. Loading into memory.")
        return faiss.read_index(index_path), np.load(ids_path)

    # 3. If it doesn't exist anywhere, create a new one
    print(f"No index found locally or in Azure for '{event_slug}'. Creating a new, empty index.")
    index = faiss.IndexFlatIP(dim) if metric == "cosine" else faiss.IndexFlatL2(dim)
    return index, np.array([], dtype=np.int64)

def persist_index(index, ids, event_slug):
    """Saves the index and IDs to the local filesystem cache."""
    os.makedirs(os.path.dirname(_idx(event_slug)), exist_ok=True)
    faiss.write_index(index, _idx(event_slug))
    np.save(_ids(event_slug), ids)
    print(f"Persisted index for '{event_slug}' to local cache.")

def add_embeddings(index, ids, embs, new_ids, metric=None):
    metric = metric or settings.FAISS_METRIC
    if metric == "cosine":
        faiss.normalize_L2(embs)
    index.add(embs.astype(np.float32))
    ids = np.concatenate([ids, new_ids.astype(np.int64)]) if ids.size else new_ids.astype(np.int64)
    return index, ids

def search(index, q, top_k=None, metric=None):
    top_k = top_k or settings.TOP_K
    metric = metric or settings.FAISS_METRIC
    q = q.astype(np.float32).reshape(1, -1)
    if metric == "cosine":
        faiss.normalize_L2(q)
    distances, indices = index.search(q, top_k)
    
    # For cosine similarity (IndexFlatIP), higher scores (closer to 1) are better.
    # For L2 distance, lower scores (closer to 0) are better. We can return similarity for both.
    if metric == "cosine":
        return distances[0], indices[0]
    else: # L2 distance, convert to a pseudo-similarity where higher is better.
        # This simple inversion isn't a true similarity score but works for ranking.
        return (1 / (1 + distances[0])), indices[0]