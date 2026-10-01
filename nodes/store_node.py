"""
Node: Qdrant Storage & Deduplication.
Stores structured jobs (matching the 35-field schema) into a dedicated
Qdrant collection per project, preventing duplicate insertions.
"""
from __future__ import annotations

import os
import hashlib
import re
from typing import Dict, Any, List, Optional
from qdrant_client import QdrantClient, models
from fastembed import SparseTextEmbedding
import ollama

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or None
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "embeddinggemma:300m")
SPARSE_MODEL = "Qdrant/bm25"

_client: Optional[QdrantClient] = None
_sparse_model: Optional[SparseTextEmbedding] = None


def get_qdrant_client() -> QdrantClient:
    """Return singleton QdrantClient instance supporting both Cloud and Localhost."""
    global _client
    if _client is None:
        url = os.getenv("QDRANT_URL", "http://localhost:6333")
        api_key = os.getenv("QDRANT_API_KEY") or None
        _client = QdrantClient(url=url, api_key=api_key, check_compatibility=False)
    return _client


def get_sparse_model() -> SparseTextEmbedding:
    """Return singleton FastEmbed SparseTextEmbedding instance."""
    global _sparse_model
    if _sparse_model is None:
        _sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL)
    return _sparse_model


def make_job_id(job: Dict[str, Any]) -> str:
    """Create a stable deterministic MD5 hash ID for deduplication."""
    url = str(job.get("sourceUrl") or "").strip().lower()
    if url and url != "https://www.linkedin.com/jobs":
        raw = url
    else:
        title = str(job.get("jobTitle") or "").strip().lower()
        company = str(job.get("company") or "").strip().lower()
        city = str(job.get("city") or "").strip().lower()
        raw = f"{title}|{company}|{city}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def ensure_project_collection(client: QdrantClient, collection_name: str, dense_dim: int = 512):
    """Ensure the project's Qdrant collection exists with dense (512D MRL) and sparse vector configurations."""
    if client.collection_exists(collection_name):
        return

    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "dense": models.VectorParams(
                size=dense_dim,
                distance=models.Distance.COSINE,
            )
        },
        sparse_vectors_config={
            "sparse": models.SparseVectorParams()
        },
    )

    # Create payload indexes on filterable fields
    keyword_fields = [
        "jobTitle", "company", "industry", "city", "state", "country",
        "experienceLevel", "jobCategory", "verifiedStatus",
        "remoteHybrid", "wfh", "onsite"
    ]
    for field in keyword_fields:
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field,
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
        except Exception:
            pass

    numeric_fields = ["salaryMin", "salaryMax", "minExperienceYears"]
    for field in numeric_fields:
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field,
                field_schema=models.PayloadSchemaType.FLOAT,
            )
        except Exception:
            pass


def store_jobs_to_project_collection(
    jobs: List[Dict[str, Any]],
    collection_name: str,
) -> Dict[str, Any]:
    """
    Ingest a list of structured jobs into the project's Qdrant collection.
    Performs duplicate prevention by checking existing IDs.
    Returns: { "stored_count": int, "skipped_count": int, "total_jobs": int }
    """
    if not jobs:
        return {"stored_count": 0, "skipped_count": 0, "total_jobs": 0}

    client = get_qdrant_client()

    # 1. Identify IDs and perform deduplication
    job_entries = []
    job_ids = []
    for j in jobs:
        jid = make_job_id(j)
        job_entries.append((jid, j))
        job_ids.append(jid)

    # Check which IDs already exist in collection
    existing_ids = set()
    if client.collection_exists(collection_name):
        try:
            existing_points = client.retrieve(
                collection_name=collection_name,
                ids=job_ids,
                with_payload=False,
                with_vectors=False,
            )
            for ep in existing_points:
                existing_ids.add(str(ep.id))
        except Exception as e:
            print(f"[INFO] Collection retrieve check: {e}")

    # Filter out duplicates
    new_entries = [(jid, j) for jid, j in job_entries if jid not in existing_ids]
    skipped_count = len(job_entries) - len(new_entries)

    if not new_entries:
        try:
            total_count = client.count(collection_name=collection_name).count
        except Exception:
            total_count = 0
        return {
            "stored_count": 0,
            "skipped_count": skipped_count,
            "total_jobs": total_count,
        }

    # 2. Dense Embeddings on jobSummary ONLY, truncated to 512 dimensions (Matryoshka Representation Learning)
    texts = [str(j.get("jobSummary") or j.get("Job Summary") or "").strip() for _, j in new_entries]

    dense_embeddings = []
    try:
        res = ollama.embed(model=EMBEDDING_MODEL, input=texts)
        dense_embeddings = [emb[:512] for emb in res["embeddings"]]
    except Exception as e:
        print(f"[WARN] Dense embedding generation via {EMBEDDING_MODEL} failed: {e}")
        dense_embeddings = [[0.0] * 512 for _ in texts]

    dense_dim = 512
    ensure_project_collection(client, collection_name, dense_dim=dense_dim)

    # Generate Sparse Embeddings via FastEmbed BM25
    sparse_model = get_sparse_model()
    sparse_embeddings = list(sparse_model.embed(texts))

    # 3. Create Qdrant Points
    points = []
    for idx, (jid, job) in enumerate(new_entries):
        dense_vec = dense_embeddings[idx]
        sparse_vec = sparse_embeddings[idx]

        payload = dict(job)
        payload["job_id"] = jid
        payload["embedding_text"] = texts[idx]

        points.append(
            models.PointStruct(
                id=jid,
                vector={
                    "dense": dense_vec,
                    "sparse": models.SparseVector(
                        indices=sparse_vec.indices.tolist(),
                        values=sparse_vec.values.tolist(),
                    ),
                },
                payload=payload,
            )
        )

    # 4. Upsert Points in batches
    batch_size = 64
    for i in range(0, len(points), batch_size):
        client.upsert(
            collection_name=collection_name,
            points=points[i:i + batch_size],
        )

    total_count = client.count(collection_name=collection_name).count

    return {
        "stored_count": len(points),
        "skipped_count": skipped_count,
        "total_jobs": total_count,
    }


# Forward and backward compatibility aliases
store_jobs_to_collection = store_jobs_to_project_collection
ensure_collection = ensure_project_collection

