"""
Legacy compatibility wrapper.
All search algorithms, prompts, and retrieval nodes have been consolidated
into `nodes/retrieval_node.py`.
"""
from nodes.retrieval_node import (
    search_jobs,
    format_jobs_for_llm,
    ALL_SCHEMA_COLUMNS,
    ALL_EXCEL_COLUMNS,
    RETRIEVAL_TOP_K,
    RERANK_TOP_K,
    FINAL_TOP_K,
    DEFAULT_MODEL as LLM_MODEL,
    EMBEDDING_MODEL,
    SPARSE_MODEL,
    RERANKER_MODEL,
    get_qdrant_client as get_client,
    get_sparse_model,
    get_reranker,
)

__all__ = [
    "search_jobs",
    "format_jobs_for_llm",
    "ALL_SCHEMA_COLUMNS",
    "ALL_EXCEL_COLUMNS",
    "RETRIEVAL_TOP_K",
    "RERANK_TOP_K",
    "FINAL_TOP_K",
    "LLM_MODEL",
    "EMBEDDING_MODEL",
    "SPARSE_MODEL",
    "RERANKER_MODEL",
    "get_client",
    "get_sparse_model",
    "get_reranker",
]