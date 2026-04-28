import os
import faiss
import joblib
import numpy as np
from typing import List, Dict, Any

from app.observability.metrics import RETRIEVAL_COUNT


ARTIFACT_DIR = os.getenv("ARTIFACT_DIR", "artifacts")


class FaissRetriever:
    def __init__(self):
        catalog_path = os.path.join(ARTIFACT_DIR, "catalog.pkl")
        vectorizer_path = os.path.join(ARTIFACT_DIR, "vectorizer.pkl")
        index_path = os.path.join(ARTIFACT_DIR, "faiss.index")

        if not os.path.exists(catalog_path):
            raise FileNotFoundError(f"Missing artifact: {catalog_path}")

        if not os.path.exists(vectorizer_path):
            raise FileNotFoundError(f"Missing artifact: {vectorizer_path}")

        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Missing artifact: {index_path}")

        self.catalog = joblib.load(catalog_path)
        self.vectorizer = joblib.load(vectorizer_path)
        self.index = faiss.read_index(index_path)

    def query(self, text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        RETRIEVAL_COUNT.inc()

        vector = self.vectorizer.transform([text]).toarray().astype("float32")

        distances, indices = self.index.search(vector, top_k)

        results = []

        for rank, idx in enumerate(indices[0], start=1):
            if idx < 0:
                continue

            row = self.catalog.iloc[idx].to_dict()

            results.append(
                {
                    "rank": rank,
                    "content_id": int(row["content_id"]),
                    "title": row["title"],
                    "genre": row["genre"],
                    "description": row["description"],
                    "year": int(row["year"]),
                    "distance": float(distances[0][rank - 1]),
                    "score": float(1 / (1 + distances[0][rank - 1])),
                }
            )

        return results