from dataclasses import asdict
from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer
from .wiki_loader import Document


class SectionRetriever:
    """Embeds section documents and retrieves the closest sections for a query."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.documents: List[Document] = []
        self.document_embeddings = None

    def build_index(self, documents: List[Document]) -> None:
        if not documents:
            raise ValueError("No documents were provided to the retriever.")

        self.documents = documents
        texts = [doc.text for doc in documents]
        self.document_embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

    def retrieve(self, query: str, top_k: int = 4) -> List[Dict]:
        if self.document_embeddings is None:
            raise RuntimeError("Call build_index(documents) before retrieve(query).")

        query_embedding = self.model.encode([query], normalize_embeddings=True)[0]
        scores = np.matmul(self.document_embeddings, query_embedding)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(top_indices, start=1):
            doc = self.documents[int(idx)]
            results.append({
                "rank": rank,
                "score": float(scores[idx]),
                "document": asdict(doc),
            })
        return results
