"""
Vector Store Adapter - Abstract interface for vector database backends.
Allows easy switching between FAISS, Chroma, and other vector stores.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pathlib import Path


class VectorStoreAdapter(ABC):
    """
    Abstract base class for vector store implementations.
    Provides a unified interface for different vector database backends.
    """

    @abstractmethod
    def create_index(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """
        Create a new vector index with the provided texts and embeddings.

        Args:
            texts: List of text chunks
            embeddings: List of embedding vectors
            metadatas: Optional metadata for each text chunk
        """
        pass

    @abstractmethod
    def save_index(self, path: Path) -> None:
        """
        Persist the vector index to disk.

        Args:
            path: Directory path to save the index
        """
        pass

    @abstractmethod
    def load_index(self, path: Path) -> None:
        """
        Load a vector index from disk.

        Args:
            path: Directory path containing the saved index
        """
        pass

    @abstractmethod
    def similarity_search(
        self,
        query_embedding: List[float],
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform similarity search to find most relevant chunks.

        Args:
            query_embedding: Query vector embedding
            k: Number of results to return
            filter: Optional metadata filter

        Returns:
            List of dicts with 'text', 'score', and 'metadata' keys
        """
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.

        Returns:
            Dictionary with stats like num_vectors, dimension, etc.
        """
        pass


class FAISSAdapter(VectorStoreAdapter):
    """
    FAISS vector store implementation.
    Facebook AI Similarity Search - optimized for fast similarity search.
    """

    def __init__(self, embedding_dimension: int = 1536):
        """
        Initialize FAISS adapter.

        Args:
            embedding_dimension: Dimension of embedding vectors
        """
        import faiss
        import numpy as np

        self.dimension = embedding_dimension
        self.index = faiss.IndexFlatL2(embedding_dimension)
        self.texts: List[str] = []
        self.metadatas: List[Dict[str, Any]] = []
        self.faiss = faiss
        self.np = np

    def create_index(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Create FAISS index from texts and embeddings."""
        # Convert embeddings to numpy array
        embeddings_array = self.np.array(embeddings).astype("float32")

        # Add vectors to FAISS index
        self.index.add(embeddings_array)

        # Store texts and metadata
        self.texts = texts
        self.metadatas = metadatas or [{} for _ in texts]

        print(f"✅ Created FAISS index with {len(texts)} vectors")

    def save_index(self, path: Path) -> None:
        """Save FAISS index and metadata to disk."""
        import pickle

        path.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        index_path = path / "faiss.index"
        self.faiss.write_index(self.index, str(index_path))

        # Save texts and metadata
        data_path = path / "data.pkl"
        with open(data_path, "wb") as f:
            pickle.dump({"texts": self.texts, "metadatas": self.metadatas}, f)

        print(f"✅ Saved FAISS index to {path}")

    def load_index(self, path: Path) -> None:
        """Load FAISS index and metadata from disk."""
        import pickle

        # Validate index files exist
        index_path = path / "faiss.index"
        data_path = path / "data.pkl"
        
        if not index_path.exists():
            raise FileNotFoundError(
                f"FAISS index file not found: {index_path}\n"
                f"Please run build_index.py to create the index first."
            )
        
        if not data_path.exists():
            raise FileNotFoundError(
                f"Index metadata file not found: {data_path}\n"
                f"Index may be corrupted. Please rebuild with build_index.py"
            )

        # Load FAISS index
        try:
            self.index = self.faiss.read_index(str(index_path))
        except Exception as e:
            raise RuntimeError(
                f"Failed to load FAISS index: {e}\n"
                f"Index may be corrupted. Please rebuild with build_index.py"
            )

        # Load texts and metadata
        with open(data_path, "rb") as f:
            data = pickle.load(f)
            self.texts = data["texts"]
            self.metadatas = data["metadatas"]

        # Success message handled by caller

    def similarity_search(
        self,
        query_embedding: List[float],
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Perform k-NN search using FAISS."""
        # Convert query to numpy array
        query_array = self.np.array([query_embedding]).astype("float32")

        # Search
        distances, indices = self.index.search(query_array, k)

        # Build results
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.texts):  # Valid index
                result = {
                    "text": self.texts[idx],
                    "score": float(1 / (1 + dist)),  # Convert distance to similarity
                    "metadata": self.metadatas[idx],
                    "rank": i + 1,
                }

                # Apply metadata filter if provided
                if filter:
                    if all(
                        result["metadata"].get(k) == v for k, v in filter.items()
                    ):
                        results.append(result)
                else:
                    results.append(result)

        return results

    def get_stats(self) -> Dict[str, Any]:
        """Get FAISS index statistics."""
        return {
            "num_vectors": self.index.ntotal,
            "dimension": self.dimension,
            "index_type": "FAISS IndexFlatL2",
            "num_texts": len(self.texts),
        }


class ChromaAdapter(VectorStoreAdapter):
    """
    Chroma vector store implementation.
    Modern, developer-friendly vector database (for future migration).
    """

    def __init__(self, collection_name: str = "faq_collection"):
        """
        Initialize Chroma adapter.

        Args:
            collection_name: Name of the collection
        """
        raise NotImplementedError(
            "Chroma adapter not yet implemented. Use FAISSAdapter for now."
        )
        # Future implementation:
        # import chromadb
        # self.client = chromadb.Client()
        # self.collection = self.client.create_collection(collection_name)

    def create_index(self, texts, embeddings, metadatas=None):
        raise NotImplementedError("Chroma adapter not yet implemented")

    def save_index(self, path):
        raise NotImplementedError("Chroma adapter not yet implemented")

    def load_index(self, path):
        raise NotImplementedError("Chroma adapter not yet implemented")

    def similarity_search(self, query_embedding, k=5, filter=None):
        raise NotImplementedError("Chroma adapter not yet implemented")

    def get_stats(self):
        raise NotImplementedError("Chroma adapter not yet implemented")


def create_vector_store(store_type: str = "faiss", **kwargs) -> VectorStoreAdapter:
    """
    Factory function to create vector store adapter.

    Args:
        store_type: Type of vector store ('faiss' or 'chroma')
        **kwargs: Additional arguments for the adapter

    Returns:
        VectorStoreAdapter instance
    """
    if store_type.lower() == "faiss":
        return FAISSAdapter(**kwargs)
    elif store_type.lower() == "chroma":
        return ChromaAdapter(**kwargs)
    else:
        raise ValueError(f"Unknown vector store type: {store_type}")
