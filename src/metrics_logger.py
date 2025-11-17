"""
Metrics Logger for RAG System
Logs per-run metrics: tokens, latency, and estimated cost.
"""

import os
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from functools import wraps


class MetricsLogger:
    """
    Track and log metrics for LLM operations.
    Logs tokens (prompt/completion/total), latency_ms, and estimated_cost_usd.
    """
    
    # OpenAI Pricing (per 1M tokens) - Updated Nov 2025
    PRICING = {
        # GPT-5 Series
        "gpt-5.1": {"input": 1.25, "output": 10.00},
        "gpt-5": {"input": 1.25, "output": 10.00},
        "gpt-5-mini": {"input": 0.25, "output": 2.00},
        "gpt-5-nano": {"input": 0.05, "output": 0.40},
        "gpt-5.1-chat-latest": {"input": 1.25, "output": 10.00},
        "gpt-5-chat-latest": {"input": 1.25, "output": 10.00},
        "gpt-5.1-codex": {"input": 1.25, "output": 10.00},
        "gpt-5-codex": {"input": 1.25, "output": 10.00},
        "gpt-5-pro": {"input": 15.00, "output": 120.00},
        "gpt-5.1-codex-mini": {"input": 0.25, "output": 2.00},
        "gpt-5-search-api": {"input": 1.25, "output": 10.00},
        # GPT-4.1 Series
        "gpt-4.1": {"input": 2.00, "output": 8.00},
        "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
        "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
        # GPT-4o Series
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4o-2024-05-13": {"input": 5.00, "output": 15.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-4o-audio-preview": {"input": 2.50, "output": 10.00},
        "gpt-4o-mini-audio-preview": {"input": 0.15, "output": 0.60},
        "gpt-4o-realtime-preview": {"input": 5.00, "output": 20.00},
        "gpt-4o-mini-realtime-preview": {"input": 0.60, "output": 2.40},
        "gpt-4o-search-preview": {"input": 2.50, "output": 10.00},
        "gpt-4o-mini-search-preview": {"input": 0.15, "output": 0.60},
        # GPT Realtime/Audio
        "gpt-realtime": {"input": 4.00, "output": 16.00},
        "gpt-realtime-mini": {"input": 0.60, "output": 2.40},
        "gpt-audio": {"input": 2.50, "output": 10.00},
        "gpt-audio-mini": {"input": 0.60, "output": 2.40},
        # O-series (Reasoning)
        "o1": {"input": 15.00, "output": 60.00},
        "o1-pro": {"input": 150.00, "output": 600.00},
        "o1-mini": {"input": 1.10, "output": 4.40},
        "o3": {"input": 2.00, "output": 8.00},
        "o3-pro": {"input": 20.00, "output": 80.00},
        "o3-mini": {"input": 1.10, "output": 4.40},
        "o3-deep-research": {"input": 10.00, "output": 40.00},
        "o4-mini": {"input": 1.10, "output": 4.40},
        "o4-mini-deep-research": {"input": 2.00, "output": 8.00},
        # Legacy Models
        "gpt-4-turbo": {"input": 10.00, "output": 30.00},
        "gpt-4": {"input": 30.00, "output": 60.00},
        "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
        # Embeddings
        "text-embedding-3-small": {"input": 0.02, "output": 0.0},
        "text-embedding-3-large": {"input": 0.13, "output": 0.0},
        # Image Models
        "gpt-image-1": {"input": 5.00, "output": 0.0},
        "gpt-image-1-mini": {"input": 2.00, "output": 0.0},
        # Other
        "codex-mini-latest": {"input": 1.50, "output": 6.00},
        "computer-use-preview": {"input": 3.00, "output": 12.00},
    }
    
    def __init__(self, log_dir: str = "logs", verbose: bool = False):
        """
        Initialize metrics logger.
        
        Args:
            log_dir: Directory for log files
            verbose: If True, log full content; if False, log summaries
        """
        self.log_dir = Path(log_dir)
        self.verbose = verbose
        self.enabled = os.getenv("ENABLE_METRICS", "false").lower() == "true"
        
        # Create log directory if metrics enabled
        if self.enabled:
            self.log_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_log_file(self) -> Path:
        """Get log file path with daily rotation."""
        date_str = datetime.now().strftime("%Y-%m-%d")
        return self.log_dir / f"metrics_{date_str}.jsonl"
    
    @staticmethod
    def calculate_cost(tokens: Dict[str, int], model: str) -> float:
        """
        Calculate estimated cost in USD.
        
        Args:
            tokens: Dict with 'prompt' and 'completion' token counts
            model: Model name
            
        Returns:
            Estimated cost in USD
        """
        if model not in MetricsLogger.PRICING:
            return 0.0
        
        pricing = MetricsLogger.PRICING[model]
        prompt_cost = (tokens.get("prompt", 0) / 1_000_000) * pricing["input"]
        completion_cost = (tokens.get("completion", 0) / 1_000_000) * pricing["output"]
        
        return prompt_cost + completion_cost
    
    def prepare_log_entry(
        self,
        operation: str,
        question: str,
        answer: str,
        chunks: List[Dict[str, Any]],
        tokens: Dict[str, int],
        latency_ms: float,
        model: str,
    ) -> Dict[str, Any]:
        """
        Prepare log entry with configurable verbosity.
        
        Args:
            operation: Type of operation (query, evaluate)
            question: User question
            answer: System answer
            chunks: Retrieved chunks
            tokens: Token usage dict
            latency_ms: Latency in milliseconds
            model: Model name
            
        Returns:
            Log entry dict
        """
        cost_usd = self.calculate_cost(tokens, model)
        
        base_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "operation": operation,
            "question": question,  # Always log full question
            "tokens": tokens,
            "latency_ms": round(latency_ms, 2),
            "cost_usd": round(cost_usd, 6),
            "model": model,
        }
        
        if self.verbose:
            # Verbose mode: log everything
            base_entry.update({
                "answer": answer,
                "chunks": chunks,
            })
        else:
            # Summary mode: log previews
            preview_len = int(os.getenv("METRICS_ANSWER_PREVIEW_LENGTH", "100"))
            chunk_preview_len = int(os.getenv("METRICS_CHUNK_PREVIEW_LENGTH", "80"))
            
            base_entry.update({
                "answer_preview": (
                    answer[:preview_len] + ("..." if len(answer) > preview_len else "")
                ),
                "answer_length": len(answer),
                "chunks_count": len(chunks),
                "chunks_summary": [
                    {
                        "rank": chunk.get("rank", i + 1),
                        "score": chunk.get("relevance_score", 0.0),
                        "length": len(chunk.get("text", "")),
                        "preview": (
                            chunk.get("text", "")[:chunk_preview_len] + "..."
                            if len(chunk.get("text", "")) > chunk_preview_len
                            else chunk.get("text", "")
                        ),
                    }
                    for i, chunk in enumerate(chunks[:3])  # Top 3 chunks only
                ],
            })
        
        return base_entry
    
    def log(
        self,
        operation: str,
        question: str,
        answer: str,
        chunks: List[Dict[str, Any]],
        tokens: Dict[str, int],
        latency_ms: float,
        model: str,
    ):
        """
        Log metrics to JSONL file.
        
        Args:
            operation: Type of operation
            question: User question
            answer: System answer
            chunks: Retrieved chunks
            tokens: Token usage
            latency_ms: Latency in milliseconds
            model: Model name
        """
        if not self.enabled:
            return
        
        entry = self.prepare_log_entry(
            operation=operation,
            question=question,
            answer=answer,
            chunks=chunks,
            tokens=tokens,
            latency_ms=latency_ms,
            model=model,
        )
        
        log_file = self._get_log_file()
        
        # Write as JSON line (automatically escapes special characters)
        line = json.dumps(entry, ensure_ascii=False)
        
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    
    def log_evaluation(
        self,
        question: str,
        answer: str,
        evaluation: Dict[str, Any],
        tokens: Dict[str, int],
        latency_ms: float,
        model: str,
    ):
        """
        Log evaluation metrics.
        
        Args:
            question: User question
            answer: System answer
            evaluation: Evaluation result
            tokens: Token usage
            latency_ms: Latency in milliseconds
            model: Model name
        """
        if not self.enabled:
            return
        
        cost_usd = self.calculate_cost(tokens, model)
        
        entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "operation": "evaluate",
            "question": question,
            "answer_preview": answer[:100] + ("..." if len(answer) > 100 else ""),
            "overall_score": evaluation.get("overall_score", 0),
            "scores": evaluation.get("scores", {}),
            "tokens": tokens,
            "latency_ms": round(latency_ms, 2),
            "cost_usd": round(cost_usd, 6),
            "model": model,
        }
        
        if self.verbose:
            entry["answer"] = answer
            entry["evaluation_full"] = evaluation
        
        log_file = self._get_log_file()
        line = json.dumps(entry, ensure_ascii=False)
        
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def track_metrics(operation: str):
    """
    Decorator to track metrics for a function.
    
    Usage:
        @track_metrics("query")
        def query(self, question: str):
            ...
    """
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # Only track if metrics enabled
            if not os.getenv("ENABLE_METRICS", "false").lower() == "true":
                return func(self, *args, **kwargs)
            
            start_time = time.time()
            result = func(self, *args, **kwargs)
            latency_ms = (time.time() - start_time) * 1000
            
            # Extract metrics from result
            # This assumes result has certain structure - adapt as needed
            if hasattr(self, 'metrics_logger'):
                self.metrics_logger.log(
                    operation=operation,
                    question=args[0] if args else kwargs.get('question', ''),
                    answer=result.get('system_answer', ''),
                    chunks=result.get('chunks_related', []),
                    tokens=result.get('tokens', {'prompt': 0, 'completion': 0, 'total': 0}),
                    latency_ms=latency_ms,
                    model=self.llm_model,
                )
            
            return result
        return wrapper
    return decorator
