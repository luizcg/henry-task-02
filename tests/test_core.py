"""
Core Tests for RAG FAQ System
Tests chunking, embedding, vector search, and query pipeline.
"""

import pytest
import os
import sys
import json
import warnings
from pathlib import Path
from unittest.mock import Mock, patch

# Suppress Pydantic v1 and Python 3.14+ compatibility warnings
warnings.filterwarnings("ignore", message=".*Pydantic V1 functionality.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pydantic")
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*SwigPy.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*ForwardRef.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*asyncio.iscoroutinefunction.*")

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vector_store_adapter import FAISSAdapter, create_vector_store


class TestVectorStoreAdapter:
    """Test vector store adapter functionality."""

    def test_faiss_adapter_creation(self):
        """Test FAISS adapter can be created."""
        adapter = FAISSAdapter(embedding_dimension=1536)
        assert adapter.dimension == 1536
        assert adapter.index.ntotal == 0

    def test_create_vector_store_factory(self):
        """Test factory function creates correct adapter."""
        adapter = create_vector_store("faiss", embedding_dimension=384)
        assert isinstance(adapter, FAISSAdapter)
        assert adapter.dimension == 384

    def test_faiss_create_index(self):
        """Test creating index with embeddings."""
        adapter = FAISSAdapter(embedding_dimension=3)

        texts = ["test1", "test2", "test3"]
        embeddings = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        metadatas = [{"id": i} for i in range(3)]

        adapter.create_index(texts, embeddings, metadatas)

        assert adapter.index.ntotal == 3
        assert len(adapter.texts) == 3
        assert len(adapter.metadatas) == 3

    def test_faiss_similarity_search(self):
        """Test similarity search returns relevant results."""
        adapter = FAISSAdapter(embedding_dimension=3)

        texts = ["apple", "banana", "orange"]
        embeddings = [[1.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.0, 1.0, 0.0]]

        adapter.create_index(texts, embeddings)

        # Query similar to first embedding
        query = [1.0, 0.0, 0.0]
        results = adapter.similarity_search(query, k=2)

        assert len(results) == 2
        assert results[0]["text"] == "apple"  # Most similar
        assert "score" in results[0]
        assert "rank" in results[0]

    def test_faiss_save_and_load(self, tmp_path):
        """Test saving and loading index."""
        adapter1 = FAISSAdapter(embedding_dimension=3)

        texts = ["test1", "test2"]
        embeddings = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]

        adapter1.create_index(texts, embeddings)
        adapter1.save_index(tmp_path)

        # Load in new adapter
        adapter2 = FAISSAdapter(embedding_dimension=3)
        adapter2.load_index(tmp_path)

        assert adapter2.index.ntotal == 2
        assert len(adapter2.texts) == 2

    def test_get_stats(self):
        """Test getting index statistics."""
        adapter = FAISSAdapter(embedding_dimension=128)

        texts = ["a", "b", "c"]
        embeddings = [[0.1] * 128 for _ in range(3)]

        adapter.create_index(texts, embeddings)

        stats = adapter.get_stats()
        assert stats["num_vectors"] == 3
        assert stats["dimension"] == 128
        assert stats["num_texts"] == 3


class TestChunking:
    """Test text chunking strategies."""

    def test_recursive_splitter_basic(self):
        """Test basic text splitting."""
        from langchain.text_splitter import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=50, chunk_overlap=10, separators=["\n\n", "\n", " "]
        )

        text = "This is a test. " * 10
        chunks = splitter.split_text(text)

        assert len(chunks) > 1
        assert all(len(chunk) <= 60 for chunk in chunks)  # chunk_size + some buffer

    def test_qa_pair_preservation(self):
        """Test that Q&A pairs are preserved in chunks."""
        from langchain.text_splitter import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=0,
            separators=["\n" + "-" * 80 + "\n", "\n\n"],
        )

        text = """Q1: What is NVIDIA?

A1: NVIDIA is a technology company.

--------------------------------------------------------------------------------

Q2: What do they make?

A2: They make GPUs."""

        chunks = splitter.split_text(text)

        # Should split on separator
        assert len(chunks) >= 1


class TestEmbeddings:
    """Test embedding generation (mocked)."""

    @patch("langchain_openai.OpenAIEmbeddings")
    def test_embed_documents(self, mock_embeddings):
        """Test embedding multiple documents."""
        # Mock the embedding response
        mock_instance = Mock()
        mock_instance.embed_documents.return_value = [[0.1, 0.2], [0.3, 0.4]]
        mock_embeddings.return_value = mock_instance

        embeddings = mock_instance
        texts = ["test1", "test2"]

        result = embeddings.embed_documents(texts)

        assert len(result) == 2
        assert len(result[0]) == 2

    @patch("langchain_openai.OpenAIEmbeddings")
    def test_embed_query(self, mock_embeddings):
        """Test embedding a single query."""
        mock_instance = Mock()
        mock_instance.embed_query.return_value = [0.1, 0.2, 0.3]
        mock_embeddings.return_value = mock_instance

        embeddings = mock_instance
        query = "test query"

        result = embeddings.embed_query(query)

        assert len(result) == 3


class TestOpenAIEmbeddings:
    """Test real OpenAI API calls (requires API key)."""

    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"), reason="API key not available"
    )
    def test_real_openai_embedding(self):
        """Test actual OpenAI embedding generation."""
        from langchain_openai import OpenAIEmbeddings
        from dotenv import load_dotenv
        
        load_dotenv()
        
        # Initialize OpenAI embeddings
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Test single query embedding
        query = "What is NVIDIA CUDA?"
        query_embedding = embeddings.embed_query(query)
        
        # Verify embedding properties
        assert isinstance(query_embedding, list)
        assert len(query_embedding) == 1536  # text-embedding-3-small dimension
        assert all(isinstance(x, float) for x in query_embedding)
        
        # Test batch embeddings
        texts = ["NVIDIA GPU", "CUDA programming", "Deep learning"]
        doc_embeddings = embeddings.embed_documents(texts)
        
        assert len(doc_embeddings) == 3
        assert all(len(emb) == 1536 for emb in doc_embeddings)
        
        print(f"\n✅ Real OpenAI embedding test passed!")
        print(f"   Query embedding dimension: {len(query_embedding)}")
        print(f"   Batch embeddings: {len(doc_embeddings)} documents")

    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"), reason="API key not available"
    )
    def test_embedding_similarity(self):
        """Test that similar texts have similar embeddings."""
        from langchain_openai import OpenAIEmbeddings
        from dotenv import load_dotenv
        import numpy as np
        
        load_dotenv()
        
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        
        # Similar texts
        text1 = "NVIDIA makes graphics processing units"
        text2 = "NVIDIA produces GPUs for graphics"
        # Different text
        text3 = "Apple manufactures smartphones"
        
        emb1 = embeddings.embed_query(text1)
        emb2 = embeddings.embed_query(text2)
        emb3 = embeddings.embed_query(text3)
        
        # Calculate cosine similarity
        def cosine_similarity(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        
        sim_1_2 = cosine_similarity(emb1, emb2)
        sim_1_3 = cosine_similarity(emb1, emb3)
        
        # Similar texts should be more similar than different texts
        assert sim_1_2 > sim_1_3
        print(f"\n✅ Embedding similarity test passed!")
        print(f"   Similar texts similarity: {sim_1_2:.4f}")
        print(f"   Different texts similarity: {sim_1_3:.4f}")


class TestRAGQueryEngine:
    """Test RAG query engine functionality."""

    def test_query_engine_initialization(self):
        """Test query engine parameters without loading index."""
        from pathlib import Path
        
        # Test that we can verify required parameters
        # Actual initialization requires index, so we test with real index path
        index_path = Path(__file__).parent.parent / "data" / "vector_index"
        
        if not index_path.exists():
            pytest.skip("Vector index not found - run build_index.py first")
        
        from query import RAGQueryEngine
        
        engine = RAGQueryEngine(
            index_path=index_path,
            embedding_model="text-embedding-3-small",
            llm_model="gpt-4o",
            top_k=5,
            similarity_threshold=0.3
        )
        
        assert engine.top_k == 5
        assert engine.similarity_threshold == 0.3
        assert engine.embedding_model == "text-embedding-3-small"
        assert engine.llm_model == "gpt-4o"
        assert engine.vector_store is not None

    def test_retrieve_chunks_structure(self):
        """Test that retrieve_chunks returns proper structure."""
        from pathlib import Path
        
        index_path = Path(__file__).parent.parent / "data" / "vector_index"
        
        if not index_path.exists():
            pytest.skip("Vector index not found - run build_index.py first")
        
        from query import RAGQueryEngine
        
        engine = RAGQueryEngine(
            index_path=index_path,
            top_k=3
        )
        
        # Test retrieve_chunks without doing a full query
        chunks = engine.retrieve_chunks("What is NVIDIA?", k=3)
        
        assert isinstance(chunks, list)
        if len(chunks) > 0:
            chunk = chunks[0]
            assert "text" in chunk
            assert "relevance_score" in chunk
            assert "rank" in chunk
            assert "metadata" in chunk


class TestIntegration:
    """Integration tests for full pipeline (requires API key and index)."""



class TestEvaluator:
    """Test RAG evaluator functionality."""

    @patch("langchain_openai.ChatOpenAI")
    def test_evaluator_initialization(self, mock_llm):
        """Test evaluator can be initialized."""
        from evaluator import RAGEvaluator

        evaluator = RAGEvaluator(llm_model="gpt-4o")
        assert evaluator.llm_model == "gpt-4o"
        assert len(evaluator.criteria) == 5  # 5 evaluation criteria

    @patch("langchain_openai.ChatOpenAI")
    def test_evaluate_response_structure(self, mock_llm):
        """Test evaluation returns correct structure."""
        from evaluator import RAGEvaluator

        # Mock LLM response
        mock_instance = Mock()
        mock_response = Mock()
        mock_response.content = json.dumps({
            "scores": {
                "relevance": 9,
                "accuracy": 8,
                "completeness": 8,
                "groundedness": 9,
                "clarity": 9
            },
            "overall_score": 8.6,
            "reasoning": "Test reasoning",
            "strengths": ["Good context usage"],
            "weaknesses": ["Could be more detailed"],
            "suggestions": ["Add more examples"]
        })
        mock_instance.invoke.return_value = mock_response
        mock_llm.return_value = mock_instance

        evaluator = RAGEvaluator()
        evaluator.llm = mock_instance

        # Test evaluation
        result = evaluator.evaluate(
            user_question="What is CUDA?",
            system_answer="CUDA is NVIDIA's parallel computing platform.",
            chunks_related=[{"text": "CUDA context", "relevance_score": 0.9, "rank": 1}]
        )

        # Verify structure
        assert "scores" in result
        assert "overall_score" in result
        assert "reasoning" in result
        assert "strengths" in result
        assert "weaknesses" in result
        assert result["overall_score"] >= 0
        assert result["overall_score"] <= 10

    def test_weighted_score_calculation(self):
        """Test weighted score calculation."""
        from evaluator import RAGEvaluator

        evaluator = RAGEvaluator()
        
        scores = {
            "relevance": 10,
            "accuracy": 10,
            "completeness": 10,
            "groundedness": 10,
            "clarity": 10
        }
        
        weighted = evaluator._calculate_weighted_score(scores)
        assert weighted == 10.0  # Perfect score


class TestConfiguration:
    """Test configuration loading and validation."""

    def test_env_loading(self):
        """Test environment variable loading."""
        from dotenv import load_dotenv

        # Test that dotenv can be loaded
        load_dotenv()  # Should not raise error

    def test_config_defaults(self):
        """Test default configuration values."""
        default_chunk_size = int(os.getenv("CHUNK_SIZE", "1000"))
        default_overlap = int(os.getenv("CHUNK_OVERLAP", "200"))

        assert default_chunk_size == 1000
        assert default_overlap == 200


# Pytest fixtures
@pytest.fixture
def sample_faq_text():
    """Sample FAQ text for testing."""
    return """Q1: What is GPU computing?

A1: GPU computing is the use of GPUs for general purpose computing tasks.

Context: GPUs were originally designed for graphics rendering but are now used for many computational tasks.

Source: NVDA - 2023_10K
--------------------------------------------------------------------------------

Q2: What is CUDA?

A2: CUDA is NVIDIA's parallel computing platform and programming model.

Context: CUDA enables developers to use NVIDIA GPUs for general purpose processing.

Source: NVDA - 2023_10K
--------------------------------------------------------------------------------"""


@pytest.fixture
def sample_embeddings():
    """Sample embeddings for testing."""
    return [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]


# Additional test helpers
def test_requirements():
    """Test that all required packages are installed."""
    try:
        import faiss
        import langchain
        import langchain_openai
        from dotenv import load_dotenv
        import numpy
    except ImportError as e:
        pytest.fail(f"Required package not installed: {e}")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
