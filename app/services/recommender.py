import hashlib
from typing import Dict, Any, List

from app.Cache.redis_client import RedisCache
from app.retrieval.faiss_index import FaissRetriever
from app.observability.metrics import CACHE_HIT_COUNT, CACHE_MISS_COUNT


class Recommender:
    def __init__(self):
        self.retriever = FaissRetriever()
        self.cache = RedisCache()

    def _cache_key(self, query: str, top_k: int) -> str:
        raw = f"{query.lower().strip()}::{top_k}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def recommend(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        cache_key = self._cache_key(query, top_k)

        cached = self.cache.get(cache_key)

        if cached is not None:
            CACHE_HIT_COUNT.inc()
            return {
                "source": "cache",
                "query": query,
                "top_k": top_k,
                "recommendations": cached,
            }

        CACHE_MISS_COUNT.inc()

        results = self.retriever.query(query, top_k=top_k)

        self.cache.set(cache_key, results)

        return {
            "source": "fresh",
            "query": query,
            "top_k": top_k,
            "recommendations": results,
        }