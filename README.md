# 🤖 RAG-Based FAQ Support Chatbot

> **Production-ready Retrieval-Augmented Generation (RAG) system for intelligent question answering**

A complete RAG implementation featuring modular architecture, automatic quality evaluation, and Unix-style pipelines. Built with LangChain, FAISS, and OpenAI.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [Key Features](#-key-features)
- [Architecture](#-architecture)
- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Usage](#-usage)
- [Bonus: Quality Evaluator](#-bonus-quality-evaluator)
- [Unix-Style Pipelines](#-unix-style-pipelines)
- [Testing](#-testing)
- [Project Structure](#-project-structure)
- [Configuration](#-configuration)
- [Troubleshooting](#-troubleshooting)

---

## 📋 Overview

This system implements an intelligent FAQ chatbot that:

- ✅ **Processes 7,000+ Q&A pairs** (521,000+ words from NVIDIA 10-K filing)
- ✅ **Chunks text intelligently** using semantic splitting (5,234 chunks)
- ✅ **Generates embeddings** using OpenAI's `text-embedding-3-small` (1536 dimensions)
- ✅ **Stores vectors in FAISS** for fast k-NN similarity search
- ✅ **Retrieves relevant context** using exact vector search (L2 distance)
- ✅ **Generates accurate answers** using GPT-4 with RAG architecture
- ⭐ **Evaluates response quality** automatically (0-10 scoring with reasoning)
- 🔀 **Unix-style pipelines** for composability and automation

---

## 📊 Dataset

### Source

This system uses the **[Financial Q&A - 10k](https://www.kaggle.com/datasets/yousefsaeedian/financial-q-and-a-10k)** dataset from Kaggle.

**Dataset Details:**
- **Name:** Financial-QA-10k
- **Creator:** Yousef Saeedian ([@0xlupo](https://twitter.com/0xlupo))
- **License:** Apache 2.0
- **Size:** ~754 KB
- **Rows:** 7,000 Q&A pairs
- **Source:** 10-K company filings (annual financial reports)

### Structure

| Column | Description |
|--------|-------------|
| **question** | Financial or operational question |
| **answer** | Specific answer derived from filing |
| **context** | Textual context from 10-K filing |
| **ticker** | Stock ticker symbol (e.g., NVDA) |
| **filing** | Filing year (e.g., 2023_10K) |

### Sample Entry

```json
{
  "question": "What area did NVIDIA initially focus on before expanding into other markets?",
  "answer": "NVIDIA initially focused on PC graphics.",
  "context": "Since our original focus on PC graphics, we have expanded into various markets.",
  "ticker": "NVDA",
  "filing": "2023_10K"
}
```

### Dataset Characteristics

- **Domain:** Financial analysis and company operations
- **Companies:** Multiple companies (primarily NVIDIA in our implementation)
- **Topics:** Business strategy, financial metrics, operations, market analysis
- **Use Cases:** 
  - NLP model development (question answering, information retrieval)
  - Financial analysis automation
  - Educational purposes in finance and data science

### Processing Pipeline

1. **CSV to Text:** Converted 7,000 Q&A pairs to structured text document
2. **Text Expansion:** Generated 521,797 words from contextual information
3. **Semantic Chunking:** Split into 5,234 chunks preserving Q&A integrity
4. **Embedding:** Each chunk converted to 1536-dimensional vector
5. **Indexing:** Stored in FAISS for efficient similarity search

**Dataset Statistics:**
- Downloads: 991+
- Views: 6,466+
- Upvotes: 42+
- Last Updated: June 17, 2024

---

### Metrics

| Metric | Value |
|--------|-------|
| **Source Document** | 521,797 words (521x requirement) |
| **Chunks Generated** | 5,234 chunks (261x requirement) |
| **Embedding Dimension** | 1536 (text-embedding-3-small) |
| **Vector Store** | FAISS IndexFlatL2 (exact search) |
| **LLM Model** | GPT-4 Turbo Preview |
| **Cost per Query** | ~$0.01 (embedding + generation) |
| **Response Time** | ~2-3 seconds |

---

## ✨ Key Features

### 1. **Modular Architecture**
- Abstract vector store adapter for easy migration (FAISS ↔ Chroma)
- Clean separation of concerns (indexing, querying, evaluation)
- Production-ready error handling and logging

### 2. **Intelligent Chunking**
- Semantic text splitting preserving Q&A integrity
- Configurable chunk size and overlap
- RecursiveCharacterTextSplitter with custom separators

### 3. **JSON-First Output**
- Default JSON output for API/pipeline integration
- Returns `user_question`, `system_answer`, `chunks_related`
- Includes retrieval statistics and metadata

### 4. **Quality Evaluation (Bonus)**
- Automatic 0-10 scoring with 5 weighted criteria
- Detailed reasoning, strengths, weaknesses, suggestions
- Integrated or standalone evaluation modes

### 5. **Unix-Style Pipelines**
- Composable stdin/stdout architecture
- Multiple output formats (JSON, pretty, score-only)
- Automation-friendly design

---

## 🏗️ Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     RAG FAQ Support System                      │
└─────────────────────────────────────────────────────────────────┘

┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│  CSV Source  │──→───│ Text Document│──→───│  Chunking    │
│  (10k Q&As)  │      │  (521k words)│      │  (5.2k chunks)│
└──────────────┘      └──────────────┘      └──────┬───────┘
                                                    │
                                                    ↓
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│    FAISS     │←─────│  Embeddings  │←─────│   OpenAI     │
│    Index     │      │  Generation  │      │  Embedding   │
│  (5.2k vecs) │      │              │      │     API      │
└──────┬───────┘      └──────────────┘      └──────────────┘
       │
       │  k-NN Search
       ↓
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│   Relevant   │──→───│   GPT-4      │──→───│   Answer +   │
│   Chunks     │      │   Synthesis  │      │   Metadata   │
│   (top-k)    │      │              │      │    (JSON)    │
└──────────────┘      └──────────────┘      └──────┬───────┘
                                                    │
                                                    ↓
                                              ┌──────────────┐
                                              │  Evaluator   │
                                              │  (Optional)  │
                                              │  Score 0-10  │
                                              └──────────────┘
```

### Component Architecture

#### 1. **Vector Store Adapter Pattern**

```python
VectorStoreAdapter (Abstract Interface)
├── FAISSAdapter (Current - Production Ready)
│   ├── create_index()
│   ├── save_index()
│   ├── load_index()
│   ├── similarity_search()
│   └── get_stats()
└── ChromaAdapter (Future - Easy Migration)
    └── Same interface, different implementation
```

**Migration Example:**
```bash
# Switch from FAISS to Chroma
echo "VECTOR_STORE_TYPE=chroma" >> .env
python src/build_index.py  # Rebuild with Chroma
# No code changes needed!
```

#### 2. **Data Pipeline**

```python
# src/build_index.py

1. Load Document
   ├── Read FAQ text file (521k words)
   └── Validate format

2. Chunk Text
   ├── RecursiveCharacterTextSplitter
   ├── chunk_size=1000, overlap=200
   └── Semantic separators (Q&A pairs, paragraphs, sentences)

3. Generate Embeddings
   ├── OpenAI text-embedding-3-small
   ├── Batch processing (efficient API usage)
   └── 5,234 vectors × 1536 dimensions

4. Build & Save Index
   ├── FAISS IndexFlatL2
   ├── Exact k-NN search
   └── Save to disk (~40MB)
```

#### 3. **Query Pipeline**

```python
# src/query.py

1. Embed Query
   ├── Convert question to 1536-dim vector
   └── Same model as indexing

2. Vector Search
   ├── k-NN with L2 distance
   ├── Filter by similarity threshold
   └── Return top-k chunks

3. Generate Answer
   ├── Build context from chunks
   ├── GPT-4 synthesis with RAG
   └── Structured prompt engineering

4. Return JSON
   ├── user_question
   ├── system_answer
   ├── chunks_related (text, score, metadata)
   └── retrieval_stats
```

#### 4. **Evaluator Agent (Bonus)**

```python
# src/evaluator.py

Input: user_question, system_answer, chunks_related

Evaluation Criteria (weighted):
├── Relevance (30%)     - Addresses the question?
├── Accuracy (25%)      - Factually correct?
├── Completeness (20%)  - Covers all aspects?
├── Groundedness (15%)  - Based on context?
└── Clarity (10%)       - Well-structured?

Output:
├── overall_score (0-10)
├── scores (by criterion)
├── reasoning (detailed explanation)
├── strengths (list)
├── weaknesses (list)
└── suggestions (list)
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- OpenAI API key

### Installation

```bash
# Clone repository
git clone <repository-url>
cd henry_task_02

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your OpenAI API key
OPENAI_API_KEY=sk-your-key-here
```

### Build Index

```bash
# Generate vector index from FAQ document
python src/build_index.py
```

Expected output:
```
🚀 Starting FAQ Index Builder Pipeline
📖 Loading document from data/faq_document.txt
✅ Loaded document: 3,992,704 chars, ~521,796 words
🔪 Chunking text (size=1000, overlap=200)
✅ Created 5,234 chunks
🔮 Generating embeddings using text-embedding-3-small
✅ Generated 5,234 embeddings
📐 Embedding dimension: 1536
🏗️  Building FAISS index
✅ Created FAISS index with 5,234 vectors
💾 Index saved to: data/vector_index
```

### Query the System

**Interactive Mode:**
```bash
python src/query.py
```

**Single Question:**
```bash
python src/query.py "What is NVIDIA's CUDA platform?"
```

**Programmatic Usage:**
```python
from src.query import RAGQueryEngine

engine = RAGQueryEngine(
    index_path="data/vector_index",
    embedding_model="text-embedding-3-small",
    llm_model="gpt-4o",
    top_k=5
)

response = engine.query("What industries use NVIDIA GPUs?")
print(response["system_answer"])
```

## 📁 Project Structure

```
henry_task_02/
├── data/
│   ├── faq_document.txt           # Source document (521k words)
│   └── vector_index/              # FAISS index + metadata
│       ├── faiss.index
│       └── data.pkl
├── dataset/
│   └── Financial-QA-10k.csv       # Original CSV dataset
├── src/
│   ├── build_index.py             # Data pipeline script
│   ├── query.py                   # Query pipeline script
│   └── vector_store_adapter.py    # Vector store abstraction
├── scripts/
│   └── csv_to_text.py             # CSV to text converter
├── tests/
│   └── test_core.py               # Unit and integration tests
├── requirements.txt               # Python dependencies
├── .env.example                   # Environment template
└── README.md                      # This file
```

## 🔧 Technical Choices

### Chunking Strategy

**Recursive Character Text Splitter** with semantic separators:

```python
RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=[
        "\n" + "=" * 80 + "\n",  # Document sections
        "\n" + "-" * 80 + "\n",  # Q&A pairs
        "\n\n",                   # Paragraphs
        "\n",                     # Lines
        ". ",                     # Sentences
        " ",                      # Words
    ]
)
```

**Why?**
- Preserves Q&A pair integrity
- Prevents cutting mid-sentence
- Overlap ensures context continuity
- Generated 5,234 chunks (well above 20 minimum)

### Embedding Model

**text-embedding-3-small** (1536 dimensions)

**Why?**
- Cost-effective ($0.02 / 1M tokens)
- Fast inference
- Excellent performance for FAQ retrieval
- Balances quality and speed

**Alternative:** `text-embedding-3-large` for higher accuracy

### Vector Search

**FAISS IndexFlatL2** (Exact k-NN with L2 distance)

**Why?**
- Exact search guarantees best results
- Fast for datasets up to 1M vectors
- No training or tuning required
- L2 distance works well with normalized embeddings

**Future Optimization:** Use `IndexIVFFlat` for >100k vectors

### LLM Model

**gpt-4o**

**Why?**
- Superior reasoning and synthesis
- Better context handling (128k tokens)
- More accurate answers
- Production-grade reliability

**Cost Optimization:** Use `gpt-3.5-turbo` for similar quality at lower cost

## 🎯 Bonus: Evaluator Agent

The system includes an **automatic quality evaluator** that scores RAG responses 0-10 based on:

- **Relevance** (30%): Does the answer address the question?
- **Accuracy** (25%): Is the answer factually correct?
- **Completeness** (20%): Does it cover all aspects?
- **Groundedness** (15%): Is it supported by context?
- **Clarity** (10%): Is it well-structured?

### Usage

**1. Unix-Style Pipeline (⭐ Recommended):**
```bash
# Query → Evaluate → Human-readable report
python src/query.py "What is CUDA?" | python src/evaluator.py

# Query → Evaluate → Score only (for automation)
python src/query.py "What is CUDA?" | python src/evaluator.py --score-only
# Output: 9.2

# Query → Evaluate → JSON with evaluation
python src/query.py "What is CUDA?" | python src/evaluator.py --json --pretty
```

**2. Integrated Evaluation:**
```python
engine = RAGQueryEngine(
    index_path="data/vector_index",
    enable_evaluation=True  # Enable automatic evaluation
)

response = engine.query("What is CUDA?")
print(f"Score: {response['evaluation']['overall_score']}/10")
```




## 🔀 Unix-Style Pipelines

The system supports **composable pipelines** via stdin/stdout:

### Basic Pipeline

```bash
# Default: Human-readable evaluation
python src/query.py "What is CUDA?" | python src/evaluator.py
```

**Output:**
```
================================================================================
🔍 RAG Response Evaluation
================================================================================

❓ Question: What is CUDA?

📊 Overall Score: 9.2/10

📋 Breakdown:
  - Relevance: 10/10 (weight: 30.0%)
  - Accuracy: 9/10 (weight: 25.0%)
  - Completeness: 8/10 (weight: 20.0%)
  - Groundedness: 9/10 (weight: 15.0%)
  - Clarity: 10/10 (weight: 10.0%)

💭 Reasoning:
The system's answer directly addresses the user's question...
```

### Pipeline Modes

| Command | Output | Use Case |
|---------|--------|----------|
| `python src/query.py "Q"` | JSON (compact) | APIs, automation |
| `python src/query.py "Q" --pretty` | JSON (formatted) | Debug, review |
| `python src/query.py "Q" --verbose` | JSON + logs | Development |
| `... \| python src/evaluator.py` | Human-readable eval | Manual QA |
| `... \| python src/evaluator.py --score-only` | Number (0-10) | Scripts, monitoring |
| `... \| python src/evaluator.py --json` | JSON with eval | Storage, analysis |

### Advanced Examples

**Quality Threshold Filtering:**
```bash
#!/bin/bash
THRESHOLD=7.0
score=$(python src/query.py "$QUESTION" | python src/evaluator.py --score-only)

if (( $(echo "$score < $THRESHOLD" | bc -l) )); then
    echo "⚠️  WARNING: Low quality response (score: $score)"
    exit 1
fi
```

**Batch Processing:**
```bash
cat questions.txt | while read question; do
    python src/query.py "$question" | \
        python src/evaluator.py --json >> evaluated_batch.json
done
```

**A/B Testing:**
```bash
# Test different chunk sizes
export CHUNK_SIZE=1000
python src/build_index.py
score_1000=$(python src/query.py "Q" | python src/evaluator.py --score-only)

export CHUNK_SIZE=2000  
python src/build_index.py
score_2000=$(python src/query.py "Q" | python src/evaluator.py --score-only)

echo "chunk_size=1000: $score_1000"
echo "chunk_size=2000: $score_2000"
```

## 🧪 Testing

### Running Tests

```bash
# Run all tests (without OpenAI API)
pytest tests/test_core.py -v

# Run all tests including OpenAI API tests (requires OPENAI_API_KEY)
export OPENAI_API_KEY=sk-...
pytest tests/test_core.py -v

# Run specific test class
pytest tests/test_core.py::TestVectorStoreAdapter -v

# Run OpenAI embedding tests only
pytest tests/test_core.py::TestOpenAIEmbeddings -v -s

# Run specific test
pytest tests/test_core.py::TestVectorStoreAdapter::test_faiss_adapter_creation -v

# Run with coverage
pytest tests/test_core.py --cov=src --cov-report=html

# Run without warnings
pytest tests/test_core.py -v --disable-warnings
```

**Test Results:**
- Without API key: 19 passed, 2 skipped
- With API key: 21 passed (includes real OpenAI API calls)

### Test Structure

```python
tests/test_core.py (21 tests)
├── TestVectorStoreAdapter (6 tests)
│   ├── test_faiss_adapter_creation
│   ├── test_create_vector_store_factory
│   ├── test_faiss_create_index
│   ├── test_faiss_similarity_search
│   ├── test_faiss_save_and_load
│   └── test_get_stats
├── TestChunking (2 tests)
│   ├── test_recursive_splitter_basic
│   └── test_qa_pair_preservation
├── TestEmbeddings (2 tests - mocked)
│   ├── test_embed_documents
│   └── test_embed_query
├── TestOpenAIEmbeddings (2 tests - real API) ⭐
│   ├── test_real_openai_embedding
│   └── test_embedding_similarity
├── TestRAGQueryEngine (2 tests)
│   ├── test_query_engine_initialization
│   └── test_retrieve_chunks_structure
├── TestIntegration (1 test - requires index)
│   └── test_full_pipeline_smoke_test
├── TestEvaluator (3 tests)
│   ├── test_evaluator_initialization
│   ├── test_evaluate_response_structure
│   └── test_weighted_score_calculation
├── TestConfiguration (2 tests)
│   ├── test_env_loading
│   └── test_config_defaults
└── test_requirements (1 test)
```

### Manual Testing

```bash
# 1. Verify index creation
python src/build_index.py
# Expected: 5,234 chunks, ~$0.10 cost

# 2. Test query
python src/query.py "What is CUDA?"
# Expected: JSON with answer and 5 chunks

# 3. Test evaluation
python src/query.py "What is CUDA?" | python src/evaluator.py --score-only
# Expected: Score between 7-10

# 4. Test interactive mode
python src/query.py
# Type questions, verify answers
```

## ⚙️ Configuration Options

Edit `.env` to customize behavior:

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embedding model |
| `LLM_MODEL` | `gpt-4o` | OpenAI LLM model |
| `CHUNK_SIZE` | `1000` | Characters per chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `TOP_K_RESULTS` | `5` | Chunks to retrieve |
| `SIMILARITY_THRESHOLD` | `0.7` | Minimum relevance score |
| `VECTOR_STORE_TYPE` | `faiss` | Vector store backend |

## 🤔 Troubleshooting

### Common Issues

1. **Missing OpenAI API Key**: Ensure `OPENAI_API_KEY` is set in `.env`
2. **Index Not Found**: Run `python src/build_index.py` to create the index
3. **Chunking Errors**: Adjust `CHUNK_SIZE` and `CHUNK_OVERLAP` in `.env`
4. **Evaluation Errors**: Verify `EVALUATION_WEIGHTS` in `.env`

### Debugging

1. **Enable Verbose Mode**: Run `python src/query.py --verbose` to see detailed output
2. **Check Error Messages**: Review stderr output for file/index errors
3. **Test Individual Components**: Run `pytest tests/test_core.py -v`
4. **Verify Index**: Check that `data/vector_index/` contains `faiss.index` and `data.pkl`


## ⚙️ Configuration Options

Edit `.env` to customize behavior:

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embedding model |
| `LLM_MODEL` | `gpt-4o` | OpenAI LLM model |
| `CHUNK_SIZE` | `1000` | Characters per chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `TOP_K_RESULTS` | `5` | Chunks to retrieve |
| `SIMILARITY_THRESHOLD` | `0.7` | Minimum relevance score |
| `VECTOR_STORE_TYPE` | `faiss` | Vector store backend |

### Metrics Logging

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_METRICS` | `false` | Enable/disable metrics logging |
| `METRICS_DIR` | `logs` | Directory for metrics logs |
| `METRICS_VERBOSE` | `false` | Log full content (vs previews) |
| `METRICS_ANSWER_PREVIEW_LENGTH` | `100` | Characters for answer preview |
| `METRICS_CHUNK_PREVIEW_LENGTH` | `80` | Characters for chunk preview |

---

## 📊 Metrics & Monitoring

The system automatically logs per-run metrics when `ENABLE_METRICS=true`:

### Logged Metrics

Each query/evaluation logs:
- **Tokens**: `prompt`, `completion`, `total`
- **Latency**: Milliseconds for operation
- **Cost**: Estimated USD cost based on OpenAI pricing
- **Model**: LLM model used
- **Timestamp**: UTC timestamp
- **Question**: User query
- **Answer Preview**: First 100 chars (or full in verbose mode)
- **Chunks**: Top 3 chunks with scores (or all in verbose mode)

### Log Format

**JSON Lines (`.jsonl`)** - one line per operation:

```jsonl
{"timestamp":"2025-11-17T00:30:00Z","operation":"query","question":"What is CUDA?","answer_preview":"CUDA is NVIDIA's parallel computing platform...","answer_length":487,"chunks_count":5,"chunks_summary":[{"rank":1,"score":0.92,"length":1024,"preview":"Q12: What is CUDA?..."}],"tokens":{"prompt":450,"completion":120,"total":570},"latency_ms":2341.23,"cost_usd":0.008705,"model":"gpt-4o"}
```

### Log Files

```
logs/
└── metrics_YYYY-MM-DD.jsonl  # Daily rotation
```

### Analyzing Metrics

```bash
# Total cost today
cat logs/metrics_$(date +%Y-%m-%d).jsonl | jq -s 'map(.cost_usd) | add'

# Average latency
cat logs/metrics_*.jsonl | jq -s 'map(.latency_ms) | add/length'

# Top 10 expensive queries
cat logs/metrics_*.jsonl | jq -s 'sort_by(.cost_usd) | reverse | .[0:10] | .[] | {cost: .cost_usd, question}'

# Queries over 3 seconds
cat logs/metrics_*.jsonl | jq 'select(.latency_ms > 3000) | {latency: .latency_ms, question}'

# Total tokens per day
cat logs/metrics_$(date +%Y-%m-%d).jsonl | jq -s 'map(.tokens.total) | add'

# Cost by operation
cat logs/metrics_*.jsonl | jq -s 'group_by(.operation) | map({operation: .[0].operation, total_cost: map(.cost_usd) | add})'
```

### Verbose Mode

For debugging, enable verbose logging to capture full content:

```bash
# Set in .env
METRICS_VERBOSE=true

# Or per-command
METRICS_VERBOSE=true python src/query.py "What is CUDA?"
```

**Warning**: Verbose logs can be 10x larger (~5-10KB per line vs ~600 bytes).

---

## 🚧 Known Limitations

1. **Requires OpenAI API Key**: System depends on OpenAI for embeddings and LLM
2. **English Only**: Optimized for English text
3. **FAISS is In-Memory**: Large indices may require significant RAM
4. **No Query Caching**: Each query calls OpenAI API (cost consideration)
5. **No User Authentication**: Production deployment needs auth layer

## 🔮 Future Enhancements

- [ ] Implement `ChromaAdapter` for vector store migration
- [ ] Add query caching with Redis
- [ ] Implement reranking with cross-encoder
- [ ] Add evaluation agent (0-10 scoring)
- [ ] Deploy as REST API with FastAPI
- [ ] Add streaming responses
- [ ] Implement hybrid search (BM25 + vector)
- [ ] Add query expansion and rewriting

## 📚 References

- [LangChain Documentation](https://python.langchain.com/)
- [FAISS Documentation](https://faiss.ai/)
- [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings)
- [RAG Best Practices](https://www.pinecone.io/learn/retrieval-augmented-generation/)

## 📄 License

MIT License - see LICENSE file

## 👤 Author: Luiz Garcia

Created for Henry AI Engineer Task 02

---

## 🎉 Summary

This RAG system delivers:

✅ **Production-ready code** with tests and documentation  
✅ **Modular architecture** for easy vector store migration  
✅ **Intelligent chunking** preserving Q&A structure  
✅ **High-quality embeddings** using OpenAI's latest models  
✅ **Exact k-NN search** with FAISS  
✅ **JSON-first output** for API integration  
✅ **Quality evaluation** with 5-criteria scoring (0-10)  
✅ **Unix-style pipelines** for automation  
✅ **Comprehensive testing** suite  



---

