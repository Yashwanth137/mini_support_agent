import os
import pickle
import faiss
import numpy as np
from pathlib import Path
from typing import List
from sentence_transformers import SentenceTransformer
from config import settings
from models import Document

class RAGPipeline:
    def __init__(self):
        self.encoder = SentenceTransformer(settings.embedding_model_name)
        self.dimension = self.encoder.get_embedding_dimension()
        self.index = None
        self.documents: List[Document] = []
        
        self._initialize_index()

    def _initialize_index(self):
        if settings.faiss_index_path.exists() and settings.faiss_metadata_path.exists():
            print("Loading existing FAISS index...")
            self.index = faiss.read_index(str(settings.faiss_index_path))
            with open(settings.faiss_metadata_path, 'rb') as f:
                self.documents = pickle.load(f)
        else:
            print("Building new FAISS index...")
            # Use Inner Product (Cosine Similarity if normalized)
            self.index = faiss.IndexFlatIP(self.dimension)
            self._build_index()

    def _chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start = end - overlap
        return chunks

    def _normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        # Avoid division by zero
        norms[norms == 0] = 1
        return vectors / norms

    def _build_index(self):
        docs_dir = settings.docs_dir
        if not docs_dir.exists():
            print(f"Docs directory not found at {docs_dir}")
            return

        all_chunks = []
        all_docs = []

        for filepath in docs_dir.glob("*.md"):
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
                chunks = self._chunk_text(text)
                for chunk in chunks:
                    all_chunks.append(chunk)
                    all_docs.append(Document(
                        content=chunk,
                        metadata={"source": filepath.name}
                    ))

        if all_chunks:
            embeddings = self.encoder.encode(all_chunks, convert_to_numpy=True)
            embeddings = self._normalize_vectors(embeddings)
            
            self.index.add(embeddings)
            self.documents = all_docs
            
            # Save index and metadata
            faiss.write_index(self.index, str(settings.faiss_index_path))
            with open(settings.faiss_metadata_path, 'wb') as f:
                pickle.dump(self.documents, f)
            print(f"Indexed {len(all_chunks)} chunks.")
        else:
            print("No documents found to index.")

    def retrieve(self, query: str, top_k: int = None, threshold: float = None) -> List[Document]:
        """
        Retrieves top_k documents. Drops any documents with cosine similarity < threshold.
        """
        if top_k is None:
            top_k = settings.rag_top_k
        if threshold is None:
            threshold = settings.rag_threshold
            
        if not self.index or self.index.ntotal == 0:
            return []

        query_embedding = self.encoder.encode([query], convert_to_numpy=True)
        query_embedding = self._normalize_vectors(query_embedding)
        
        similarities, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for sim, idx in zip(similarities[0], indices[0]):
            if idx != -1 and idx < len(self.documents) and sim >= threshold:
                results.append(self.documents[idx])
        
        return results
