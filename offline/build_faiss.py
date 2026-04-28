import os
import joblib
import faiss
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


ARTIFACT_DIR = "artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)


def load_catalog() -> pd.DataFrame:
    data = [
        {
            "content_id": 1,
            "title": "Black Horizon",
            "genre": "Sci-Fi Thriller",
            "description": "A hacker discovers a dangerous AI system controlling city surveillance.",
            "year": 2021,
        },
        {
            "content_id": 2,
            "title": "Mind Trap",
            "genre": "Psychological Thriller",
            "description": "A therapist begins seeing the same nightmares as her patient.",
            "year": 2022,
        },
        {
            "content_id": 3,
            "title": "Galaxy War",
            "genre": "Sci-Fi Action",
            "description": "Rebels fight against an empire using advanced alien technology.",
            "year": 2019,
        },
        {
            "content_id": 4,
            "title": "Dream Code",
            "genre": "Sci-Fi Psychological Thriller",
            "description": "A scientist creates dream-sharing software that reveals dark secrets.",
            "year": 2023,
        },
        {
            "content_id": 5,
            "title": "Crime Ledger",
            "genre": "Crime Mystery",
            "description": "A detective uncovers a financial conspiracy behind multiple murders.",
            "year": 2021,
        },
        {
            "content_id": 6,
            "title": "Silent Echo",
            "genre": "Mystery Thriller",
            "description": "A journalist investigates disappearances in a remote town.",
            "year": 2022,
        },
        {
            "content_id": 7,
            "title": "Family Fun House",
            "genre": "Family Comedy",
            "description": "A chaotic family turns everyday life into comedy.",
            "year": 2018,
        },
        {
            "content_id": 8,
            "title": "Ocean Life",
            "genre": "Documentary Nature",
            "description": "An exploration of marine life, ecosystems, and climate impact.",
            "year": 2017,
        },
        {
            "content_id": 9,
            "title": "Shadow Protocol",
            "genre": "Cyber Thriller",
            "description": "An intelligence analyst discovers a hidden surveillance network embedded in consumer apps.",
            "year": 2024,
        },
        {
            "content_id": 10,
            "title": "Neon Archive",
            "genre": "Sci-Fi Mystery",
            "description": "A data archivist uncovers erased memories inside a citywide neural system.",
            "year": 2023,
        },
    ]

    return pd.DataFrame(data)


def build_text_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["combined_text"] = (
        df["title"].fillna("") + " "
        + df["genre"].fillna("") + " "
        + df["description"].fillna("") + " "
        + df["year"].astype(str)
    )

    return df


def build_vectorizer(df: pd.DataFrame):
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=5000,
        ngram_range=(1, 2),
    )

    matrix = vectorizer.fit_transform(df["combined_text"])
    dense_matrix = matrix.toarray().astype("float32")

    return vectorizer, dense_matrix


def build_faiss_index(vectors: np.ndarray):
    dimension = vectors.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(vectors)

    return index


def save_artifacts(df, vectorizer, index):
    catalog_path = os.path.join(ARTIFACT_DIR, "catalog.pkl")
    vectorizer_path = os.path.join(ARTIFACT_DIR, "vectorizer.pkl")
    index_path = os.path.join(ARTIFACT_DIR, "faiss.index")

    joblib.dump(df, catalog_path)
    joblib.dump(vectorizer, vectorizer_path)
    faiss.write_index(index, index_path)

    print("Artifacts saved:")
    print(catalog_path)
    print(vectorizer_path)
    print(index_path)


def main():
    df = load_catalog()
    df = build_text_features(df)

    vectorizer, vectors = build_vectorizer(df)
    index = build_faiss_index(vectors)

    save_artifacts(df, vectorizer, index)

    print("FAISS artifact build completed successfully.")


if __name__ == "__main__":
    main()