"""
Data Pipeline Script - Build Vector Index
Loads FAQ document, chunks text, generates embeddings, and saves to vector store.
"""

import os
import sys
import warnings
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv

# Suppress Pydantic v1 compatibility warnings for Python 3.14+
warnings.filterwarnings("ignore", message=".*Pydantic V1 functionality.*")

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))
from vector_store_adapter import create_vector_store


class FAQIndexBuilder:
    """
    Builds searchable index from FAQ document.
    Handles text chunking, embedding generation, and vector store creation.
    """

    def __init__(
        self,
        document_path: str,
        output_path: str,
        embedding_model: str = "text-embedding-3-small",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        vector_store_type: str = "faiss",
    ):
        """
        Initialize index builder.

        Args:
            document_path: Path to FAQ text document
            output_path: Path to save vector index
            embedding_model: OpenAI embedding model name
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            vector_store_type: Type of vector store ('faiss' or 'chroma')
        """
        self.document_path = Path(document_path)
        self.output_path = Path(output_path)
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.vector_store_type = vector_store_type

        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(model=embedding_model)

        # Initialize text splitter with semantic separators
        # Priority: Q&A pairs -> paragraphs -> sentences -> words
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
                "\n" + "=" * 80 + "\n",  # Document sections
                "\n" + "-" * 80 + "\n",  # Q&A pairs
                "\n\n",  # Paragraphs
                "\n",  # Lines
                ". ",  # Sentences
                " ",  # Words
                "",  # Characters
            ],
            length_function=len,
            is_separator_regex=False,
        )

    def load_document(self) -> str:
        """Load FAQ document from file."""
        print(f"📖 Loading document from {self.document_path}")

        with open(self.document_path, "r", encoding="utf-8") as f:
            content = f.read()

        word_count = len(content.split())
        print(f"✅ Loaded document: {len(content):,} chars, ~{word_count:,} words")

        return content

    def chunk_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Split text into semantic chunks.

        Args:
            text: Full document text

        Returns:
            List of chunk dicts with text and metadata
        """
        print(f"\n🔪 Chunking text (size={self.chunk_size}, overlap={self.chunk_overlap})")

        # Split text
        chunks = self.text_splitter.split_text(text)

        print(f"✅ Created {len(chunks)} chunks")

        # Add metadata to each chunk
        chunk_dicts = []
        for i, chunk_text in enumerate(chunks):
            # Extract Q&A number if present
            qa_number = None
            if chunk_text.strip().startswith("Q"):
                try:
                    qa_number = int(chunk_text.split(":")[0].replace("Q", ""))
                except:
                    pass

            chunk_dict = {
                "text": chunk_text,
                "metadata": {
                    "chunk_id": i,
                    "chunk_size": len(chunk_text),
                    "qa_number": qa_number,
                    "source": "NVIDIA_Financial_QA",
                },
            }
            chunk_dicts.append(chunk_dict)

        # Print statistics
        avg_size = sum(len(c["text"]) for c in chunk_dicts) / len(chunk_dicts)
        min_size = min(len(c["text"]) for c in chunk_dicts)
        max_size = max(len(c["text"]) for c in chunk_dicts)

        print(f"📊 Chunk statistics:")
        print(f"   - Average size: {avg_size:.0f} chars")
        print(f"   - Min size: {min_size} chars")
        print(f"   - Max size: {max_size} chars")

        return chunk_dicts

    def generate_embeddings(self, chunks: List[Dict[str, Any]]) -> List[List[float]]:
        """
        Generate embeddings for all chunks.

        Args:
            chunks: List of chunk dictionaries

        Returns:
            List of embedding vectors
        """
        print(f"\n🔮 Generating embeddings using {self.embedding_model}")

        # Extract texts
        texts = [chunk["text"] for chunk in chunks]

        # Generate embeddings in batches
        embeddings = self.embeddings.embed_documents(texts)

        print(f"✅ Generated {len(embeddings)} embeddings")
        print(f"📐 Embedding dimension: {len(embeddings[0])}")

        return embeddings

    def build_index(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]) -> None:
        """
        Build and save vector index.

        Args:
            chunks: List of chunk dictionaries
            embeddings: List of embedding vectors
        """
        print(f"\n🏗️  Building {self.vector_store_type.upper()} index")

        # Create vector store
        vector_store = create_vector_store(
            store_type=self.vector_store_type,
            embedding_dimension=len(embeddings[0]),
        )

        # Extract data
        texts = [chunk["text"] for chunk in chunks]
        metadatas = [chunk["metadata"] for chunk in chunks]

        # Create index
        vector_store.create_index(texts=texts, embeddings=embeddings, metadatas=metadatas)

        # Save index
        vector_store.save_index(self.output_path)

        # Print stats
        stats = vector_store.get_stats()
        print(f"\n📊 Index Statistics:")
        for key, value in stats.items():
            print(f"   - {key}: {value}")

    def run(self) -> None:
        """Execute full pipeline."""
        print("=" * 80)
        print("🚀 Starting FAQ Index Builder Pipeline")
        print("=" * 80)

        # Step 1: Load document
        document = self.load_document()

        # Step 2: Chunk text
        chunks = self.chunk_text(document)

        # Validate chunk count
        if len(chunks) < 20:
            print(f"⚠️  WARNING: Only {len(chunks)} chunks created (minimum: 20)")
            print("   Consider reducing chunk_size or chunk_overlap")

        # Step 3: Generate embeddings
        embeddings = self.generate_embeddings(chunks)

        # Step 4: Build and save index
        self.build_index(chunks, embeddings)

        print("\n" + "=" * 80)
        print("✅ Pipeline completed successfully!")
        print(f"💾 Index saved to: {self.output_path}")
        print("=" * 80)


def main():
    """Main entry point."""
    # Load environment variables
    load_dotenv()

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ ERROR: OPENAI_API_KEY not found in environment")
        print("   Please set it in .env file or export it")
        sys.exit(1)

    # Configuration
    base_dir = Path(__file__).parent.parent
    config = {
        "document_path": base_dir / "data" / "faq_document.txt",
        "output_path": base_dir / "data" / "vector_index",
        "embedding_model": os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        "chunk_size": int(os.getenv("CHUNK_SIZE", "1000")),
        "chunk_overlap": int(os.getenv("CHUNK_OVERLAP", "200")),
        "vector_store_type": os.getenv("VECTOR_STORE_TYPE", "faiss"),
    }

    # Validate document exists
    if not config["document_path"].exists():
        print(f"❌ ERROR: Document not found at {config['document_path']}")
        sys.exit(1)

    # Build index
    builder = FAQIndexBuilder(**config)
    builder.run()


if __name__ == "__main__":
    main()
