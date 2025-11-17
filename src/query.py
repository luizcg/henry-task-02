"""
Query Pipeline Script - RAG Query System
Accepts user questions, performs vector search, and generates answers using LLM.
"""

import os
import sys
import json
import warnings
from pathlib import Path
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Suppress Pydantic v1 compatibility warnings for Python 3.14+
warnings.filterwarnings("ignore", message=".*Pydantic V1 functionality.*")

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))
from vector_store_adapter import create_vector_store
from evaluator import RAGEvaluator

# Global verbosity flag (set by CLI args)
VERBOSE_MODE = False

def vprint(*args, **kwargs):
    """Print only if verbose mode is enabled."""
    if VERBOSE_MODE:
        print(*args, **kwargs)


class RAGQueryEngine:
    """
    RAG Query Engine for FAQ system.
    Handles question embedding, vector search, and answer generation.
    """

    def __init__(
        self,
        index_path: str,
        embedding_model: str = "text-embedding-3-small",
        llm_model: str = "gpt-4o",
        top_k: int = 5,
        similarity_threshold: float = 0.7,
        vector_store_type: str = "faiss",
        enable_evaluation: bool = False,
    ):
        """
        Initialize RAG query engine.

        Args:
            index_path: Path to vector index
            embedding_model: OpenAI embedding model name
            llm_model: OpenAI LLM model name
            top_k: Number of chunks to retrieve
            similarity_threshold: Minimum similarity score
            vector_store_type: Type of vector store
            enable_evaluation: Enable automatic quality evaluation
        """
        self.index_path = Path(index_path)
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.enable_evaluation = enable_evaluation
        self.vector_store_type = vector_store_type

        # Initialize embeddings and LLM
        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        self.llm = ChatOpenAI(model=llm_model, temperature=0.3)

        # Load vector store
        self.vector_store = self._load_vector_store()

        # Initialize evaluator if enabled
        self.evaluator = RAGEvaluator(llm_model=llm_model) if enable_evaluation else None

        # System prompt for answer generation
        self.system_prompt = """You are an expert AI assistant for NVIDIA's financial and technical information.

Your task is to answer questions based ONLY on the provided context from NVIDIA's 10-K filing.

Guidelines:
1. Answer concisely and accurately based on the context
2. If the context doesn't contain enough information, say so clearly
3. Cite specific details from the context when possible
4. Use professional, clear language
5. Do not make up or infer information beyond the context
6. If multiple context chunks provide related information, synthesize them coherently

Context will be provided as numbered chunks. Use them to formulate your answer."""

    def _load_vector_store(self):
        """Load vector store from disk."""
        vprint(f"📂 Loading vector index from {self.index_path}")
        
        vector_store = create_vector_store(
            store_type=self.vector_store_type,
            embedding_dimension=1536,  # text-embedding-3-small dimension
        )
        
        try:
            vector_store.load_index(self.index_path)
            vprint(f"✅ Loaded index: {vector_store.get_stats()['num_vectors']} vectors")
        except FileNotFoundError as e:
            print(f"\n❌ ERROR: {e}", file=sys.stderr)
            sys.exit(1)
        except RuntimeError as e:
            print(f"\n❌ ERROR: {e}", file=sys.stderr)
            sys.exit(1)
        
        return vector_store

    def embed_query(self, question: str) -> List[float]:
        """
        Generate embedding for user question.

        Args:
            question: User's question

        Returns:
            Embedding vector
        """
        return self.embeddings.embed_query(question)

    def retrieve_chunks(
        self,
        question: str,
        k: Optional[int] = None,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks using vector search.

        Args:
            question: User's question
            k: Number of chunks to retrieve (default: self.top_k)
            filter: Optional metadata filter

        Returns:
            List of relevant chunks with scores
        """
        k = k or self.top_k

        # Generate query embedding
        query_embedding = self.embed_query(question)

        # Perform similarity search
        results = self.vector_store.similarity_search(
            query_embedding=query_embedding,
            k=k,
            filter=filter,
        )

        # Debug: print raw scores
        if results:
            vprint(f"🔍 Debug: Top result score: {results[0]['score']:.4f}, threshold: {self.similarity_threshold}")
            vprint(f"   Raw distances: {[r['score'] for r in results[:3]]}")
        else:
            vprint(f"⚠️  Debug: No results returned from vector store")

        # Filter by similarity threshold
        filtered_results = [
            r for r in results if r["score"] >= self.similarity_threshold
        ]

        if not filtered_results and results:
            vprint(f"⚠️  All {len(results)} results filtered out by threshold {self.similarity_threshold}")
            vprint(f"   Consider lowering SIMILARITY_THRESHOLD in .env")

        return filtered_results

    def generate_answer(
        self, question: str, chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate answer using LLM with retrieved chunks.

        Args:
            question: User's question
            chunks: Retrieved context chunks

        Returns:
            Dict with question, answer, and chunks
        """
        # Build context from chunks
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(f"[Context {i}]")
            context_parts.append(chunk["text"])
            context_parts.append(f"[Relevance Score: {chunk['score']:.3f}]")
            context_parts.append("")

        context = "\n".join(context_parts)

        # Build prompt
        user_message = f"""Question: {question}

Context:
{context}

Please provide a comprehensive answer based on the context above."""

        # Generate answer
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=user_message),
        ]

        response = self.llm.invoke(messages)
        answer = response.content

        # Build response
        return {
            "user_question": question,
            "system_answer": answer,
            "chunks_related": [
                {
                    "text": chunk["text"],
                    "relevance_score": chunk["score"],
                    "rank": chunk["rank"],
                    "metadata": chunk["metadata"],
                }
                for chunk in chunks
            ],
            "retrieval_stats": {
                "chunks_retrieved": len(chunks),
                "top_k": self.top_k,
                "similarity_threshold": self.similarity_threshold,
                "embedding_model": self.embedding_model,
                "llm_model": self.llm_model,
            },
        }

    def query(
        self,
        question: str,
        k: Optional[int] = None,
        filter: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Process complete query: retrieve chunks and generate answer.

        Args:
            question: User's question
            k: Number of chunks to retrieve
            filter: Optional metadata filter

        Returns:
            Complete query response with answer and chunks
        """
        vprint(f"\n🔍 Processing query: '{question}'")

        # Retrieve relevant chunks
        chunks = self.retrieve_chunks(question, k=k, filter=filter)

        if not chunks:
            return {
                "user_question": question,
                "system_answer": "I couldn't find relevant information in the knowledge base to answer this question. Please try rephrasing or asking about a different topic.",
                "chunks_related": [],
                "retrieval_stats": {
                    "chunks_retrieved": 0,
                    "top_k": k or self.top_k,
                    "similarity_threshold": self.similarity_threshold,
                    "embedding_model": self.embedding_model,
                    "llm_model": self.llm_model,
                },
            }

        vprint(f"📚 Retrieved {len(chunks)} relevant chunks")

        # Generate answer
        response = self.generate_answer(question, chunks)

        vprint(f"✅ Generated answer ({len(response['system_answer'])} chars)")

        # Evaluate if enabled
        if self.evaluator:
            vprint(f"🔍 Evaluating response quality...")
            evaluation = self.evaluator.evaluate(
                user_question=question,
                system_answer=response["system_answer"],
                chunks_related=response["chunks_related"],
            )
            response["evaluation"] = evaluation
            vprint(f"📊 Quality Score: {evaluation['overall_score']}/10")

        return response

    def batch_query(self, questions: List[str]) -> List[Dict[str, Any]]:
        """
        Process multiple queries in batch.

        Args:
            questions: List of questions

        Returns:
            List of query responses
        """
        print(f"\n📊 Processing batch of {len(questions)} queries")

        responses = []
        for i, question in enumerate(questions, 1):
            print(f"\n--- Query {i}/{len(questions)} ---")
            response = self.query(question)
            responses.append(response)

        return responses


def main():
    """Main entry point for interactive queries."""
    import argparse
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Query the RAG FAQ system")
    parser.add_argument("question", nargs="*", help="Question to ask (optional, for non-interactive mode)")
    parser.add_argument("--verbose", action="store_true", help="Show processing details and logs")
    parser.add_argument("--pretty", action="store_true", help="Pretty print JSON output")
    args = parser.parse_args()
    
    # Set global verbosity flag
    global VERBOSE_MODE
    VERBOSE_MODE = args.verbose
    
    # Load environment variables
    load_dotenv()

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ ERROR: OPENAI_API_KEY not found in environment")
        sys.exit(1)

    # Configuration
    base_dir = Path(__file__).parent.parent
    config = {
        "index_path": base_dir / "data" / "vector_index",
        "embedding_model": os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        "llm_model": os.getenv("LLM_MODEL", "gpt-4o"),
        "top_k": int(os.getenv("TOP_K_RESULTS", "5")),
        "similarity_threshold": float(os.getenv("SIMILARITY_THRESHOLD", "0.7")),
        "vector_store_type": os.getenv("VECTOR_STORE_TYPE", "faiss"),
    }

    # Validate index exists
    if not config["index_path"].exists():
        print(f"❌ ERROR: Index not found at {config['index_path']}")
        print("   Run build_index.py first to create the index")
        sys.exit(1)

    # Initialize query engine (suppress startup messages unless verbose)
    if args.verbose:
        print("=" * 80)
        print("🤖 RAG Query Engine - NVIDIA Financial FAQ")
        print("=" * 80)

    engine = RAGQueryEngine(**config)

    # Command line query mode
    if args.question:
        question = " ".join(args.question)
        response = engine.query(question)

        # Output: JSON by default, pretty if requested
        indent = 2 if args.pretty else None
        print(json.dumps(response, indent=indent, ensure_ascii=False))

    else:
        # Interactive REPL
        print("=" * 80)
        print("🤖 RAG Query Engine - NVIDIA Financial FAQ")
        print("=" * 80)
        print("\n💬 Interactive mode. Type 'quit' or 'exit' to stop.\n")

        while True:
            try:
                question = input("❓ Your question: ").strip()

                if question.lower() in ["quit", "exit", "q"]:
                    print("👋 Goodbye!")
                    break

                if not question:
                    continue

                response = engine.query(question)

                print("\n" + "-" * 80)
                print(f"🤖 Answer:\n")
                print(response["system_answer"])
                print(f"\n📊 Based on {len(response['chunks_related'])} relevant chunks")
                print("-" * 80 + "\n")

            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}\n")


if __name__ == "__main__":
    main()
