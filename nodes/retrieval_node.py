"""
Node: Unified Job Retrieval & Recommendation.
Consolidates search algorithms (Dense + Sparse BM25 + RRF + CrossEncoder reranker)
and LangGraph recommendation node into a single module.
Retrieves strictly from the active project's Qdrant collection and presents all 35 schema fields.
"""
from __future__ import annotations

import os
import re
from typing import Dict, Any, List, Optional
import ollama
from qdrant_client import QdrantClient, models
from fastembed import SparseTextEmbedding

try:
    from sentence_transformers import CrossEncoder
except ImportError:
    CrossEncoder = None

from logger_util import safe_parse_json, clean_number, save_pipeline_log

DEFAULT_MODEL = "bjoernb/gemma4-31b-fast:latest"
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or None
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "embeddinggemma:300m")
SPARSE_MODEL = "Qdrant/bm25"
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "Qwen/Qwen3-Reranker-0.6B")

RETRIEVAL_TOP_K = 20
RERANK_TOP_K = 20
FINAL_TOP_K = 5

_client: Optional[QdrantClient] = None
_sparse_model: Optional[SparseTextEmbedding] = None
_reranker: Optional[Any] = None


def get_qdrant_client() -> QdrantClient:
    """Return singleton QdrantClient instance supporting both Cloud and Localhost."""
    global _client
    if _client is None:
        url = os.getenv("QDRANT_URL", "http://localhost:6333")
        api_key = os.getenv("QDRANT_API_KEY") or None
        _client = QdrantClient(url=url, api_key=api_key, check_compatibility=False)
    return _client


def get_sparse_model() -> SparseTextEmbedding:
    global _sparse_model
    if _sparse_model is None:
        _sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL)
    return _sparse_model


def _rerank_via_hf_api(query: str, candidate_texts: List[str], token: str) -> Optional[List[float]]:
    """Query Hugging Face Inference API for reranking to avoid loading heavy models in Streamlit RAM."""
    import requests
    headers = {"Authorization": f"Bearer {token}"}
    api_url = f"https://api-inference.huggingface.co/models/{RERANKER_MODEL}"
    try:
        resp = requests.post(
            api_url,
            headers=headers,
            json={"inputs": {"source_sentence": query, "sentences": candidate_texts}},
            timeout=8,
        )
        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, list):
                if data and isinstance(data[0], (int, float)):
                    return [float(x) for x in data]
                if data and isinstance(data[0], dict) and "score" in data[0]:
                    return [float(x["score"]) for x in data]
    except Exception as e:
        print(f"[INFO] HF Reranker API skipped: {e}")
    return None


def get_reranker():
    global _reranker
    # Prefer lightweight API or graceful fallback when running on Streamlit Cloud
    if os.getenv("USE_LOCAL_RERANKER", "false").lower() not in ("true", "1", "yes"):
        return "api"

    if _reranker is None and CrossEncoder:
        token = os.getenv("HF_TOKEN") 
        try:
            _reranker = CrossEncoder(RERANKER_MODEL, token=token)
        except Exception as e:
            print(f"[WARN] Failed to load local reranker {RERANKER_MODEL}: {e}")
            _reranker = None
    return _reranker


# ============================================================
# All 35 Schema Field Keys
# ============================================================

ALL_SCHEMA_COLUMNS = [
    "jobTitle",
    "company",
    "industry",
    "companySize",
    "city",
    "state",
    "country",
    "remoteHybrid",
    "wfh",
    "onsite",
    "experienceLevel",
    "minExperienceYears",
    "ExperienceYears",
    "skillsRequired",
    "salaryMin",
    "salaryMax",
    "jobCategory",
    "jobTags",
    "datePosted",
    "dateFound",
    "verifiedStatus",
    "sourceUrl",
    "companyWebsite",
    "companyLinkedIn",
    "jobSummary",
    "jobDescription",
    "companyEmail",
    "companyPhone",
    "companyFoundedHq",
    "hiringContactName",
    "hiringContactTitle",
    "hiringContactDesignation",
    "hiringContactLinkedIn",
    "hiringContactInfo",
]

# For backwards compatibility with legacy code
ALL_EXCEL_COLUMNS = ALL_SCHEMA_COLUMNS


# ============================================================
# Formatting for LLM Prompt
# ============================================================

def format_jobs_for_llm(jobs: List[Dict[str, Any]]) -> str:
    """Format all candidate jobs with complete 35-field payload data for LLM context."""
    if not jobs:
        return "No jobs currently available in memory."

    entries = []
    for rank, job in enumerate(jobs, start=1):
        lines = [f"### Candidate Job #{rank}"]
        score = job.get("score")
        if score is not None:
            score_str = f"{score:.4f}" if isinstance(score, float) else str(score)
            lines.append(f"- **Match Score**: {score_str}")

        processed = set()
        for col in ALL_SCHEMA_COLUMNS:
            val = job.get(col)
            if val is not None and str(val).strip():
                lines.append(f"- **{col}**: {val}")
                processed.add(col)

        # Include any legacy or additional metadata
        for k, v in job.items():
            if k not in processed and k not in ("rank", "score", "embedding_text", "Embedding Text") and not k.startswith("_"):
                if v is not None and str(v).strip():
                    lines.append(f"- **{k}**: {v}")

        entries.append("\n".join(lines))
    return "\n\n".join(entries)


# ============================================================
# Qdrant Hybrid Search with RRF and Reranking
# ============================================================

def build_qdrant_filter(**kwargs) -> Optional[models.Filter]:
    """Construct Qdrant Filter conditions for schema fields."""
    must_conditions = []

    # Keyword filters
    str_fields = [
        "company", "industry", "companySize", "city", "state", "country",
        "experienceLevel", "jobCategory"
    ]
    for sf in str_fields:
        val = kwargs.get(sf)
        if val:
            must_conditions.append(
                models.FieldCondition(key=sf, match=models.MatchValue(value=str(val).strip()))
            )

    # Boolean filters
    for bf in ["remoteHybrid", "wfh", "onsite", "verifiedStatus"]:
        val = kwargs.get(bf)
        if val is not None:
            bool_val = True if str(val).lower() in ("true", "1", "yes") else False
            must_conditions.append(
                models.FieldCondition(key=bf, match=models.MatchValue(value=bool_val))
            )

    # Numeric range filters
    sal_min = kwargs.get("salaryMin") or kwargs.get("salary_min")
    if sal_min is not None:
        c_val = clean_number(sal_min)
        if c_val:
            must_conditions.append(
                models.FieldCondition(key="salaryMin", range=models.Range(gte=c_val))
            )

    sal_max = kwargs.get("salaryMax") or kwargs.get("salary_max")
    if sal_max is not None:
        c_val = clean_number(sal_max)
        if c_val:
            must_conditions.append(
                models.FieldCondition(key="salaryMax", range=models.Range(lte=c_val))
            )

    exp_years = kwargs.get("minExperienceYears") or kwargs.get("experience_years")
    if exp_years is not None:
        c_val = clean_number(exp_years)
        if c_val:
            must_conditions.append(
                models.FieldCondition(key="minExperienceYears", range=models.Range(lte=c_val))
            )

    if not must_conditions:
        return None
    return models.Filter(must=must_conditions)


def reciprocal_rank_fusion(dense_pts: list, sparse_pts: list, k: int = 60) -> list:
    """Combine dense and sparse candidate rankings via Reciprocal Rank Fusion (RRF)."""
    scores: Dict[str, float] = {}
    pts_by_id: Dict[str, Any] = {}

    for rank, pt in enumerate(dense_pts):
        pid = str(pt.id)
        pts_by_id[pid] = pt
        scores[pid] = scores.get(pid, 0.0) + (1.0 / (k + rank + 1))

    for rank, pt in enumerate(sparse_pts):
        pid = str(pt.id)
        pts_by_id[pid] = pt
        scores[pid] = scores.get(pid, 0.0) + (1.0 / (k + rank + 1))

    sorted_ids = sorted(scores.keys(), key=lambda pid: scores[pid], reverse=True)
    fused = []
    for pid in sorted_ids:
        pt = pts_by_id[pid]
        pt.score = scores[pid]
        fused.append(pt)
    return fused


def rerank_candidates(query: str, points: list, top_k: int = 5) -> list:
    """Rerank candidates using Hugging Face API, local CrossEncoder, or fallback to RRF order."""
    if not points:
        return []
    reranker = get_reranker()
    if not reranker:
        return points[:top_k]

    candidates = points[:RERANK_TOP_K]
    pairs = []
    candidate_texts = []
    for pt in candidates:
        payload = pt.payload or {}
        text = str(payload.get("jobSummary") or payload.get("jobTitle") or "")[:500]
        pairs.append([query, text])
        candidate_texts.append(text)

    # 1. HF Inference API mode (lightweight, zero RAM overhead)
    if reranker == "api":
        token = os.getenv("HF_TOKEN")
        scores = _rerank_via_hf_api(query, candidate_texts, token)
        if scores and len(scores) == len(candidates):
            ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
            final_pts = []
            for pt, sc in ranked[:top_k]:
                pt.score = float(sc)
                final_pts.append(pt)
            return final_pts
        return candidates[:top_k]

    # 2. Local CrossEncoder mode
    try:
        scores = reranker.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        final_pts = []
        for pt, sc in ranked[:top_k]:
            pt.score = float(sc)
            final_pts.append(pt)
        return final_pts
    except Exception as e:
        print(f"[WARN] Local reranker error: {e}")
        return candidates[:top_k]


def search_jobs(
    query: str,
    collection_name: str = "linkedin_jobs",
    top_k: int = FINAL_TOP_K,
    **filter_kwargs,
) -> List[Dict[str, Any]]:
    """
    Search Qdrant for matching jobs strictly within the given collection.
    Executes Dense + Sparse BM25 + RRF + Cross-Encoder reranking.
    Returns list of dicts conforming to the 35 schema fields.
    """
    client = get_qdrant_client()

    if not client.collection_exists(collection_name):
        return []

    q_filter = build_qdrant_filter(**filter_kwargs)

    # 1. Dense Embedding (512D MRL)
    dense_vec = None
    try:
        res = ollama.embed(model=EMBEDDING_MODEL, input=query)
        raw_emb = res["embeddings"][0]
        dense_vec = raw_emb[:512]
    except Exception as e:
        print(f"[WARN] Dense query embedding failed: {e}")

    # 2. Sparse Embedding
    sparse_model = get_sparse_model()
    sparse_emb = list(sparse_model.embed([query]))[0]

    dense_pts = []
    if dense_vec:
        try:
            res_dense = client.query_points(
                collection_name=collection_name,
                query=dense_vec,
                using="dense",
                query_filter=q_filter,
                limit=RETRIEVAL_TOP_K,
            )
            dense_pts = getattr(res_dense, "points", []) or []
        except Exception as e:
            print(f"[WARN] Dense Qdrant search failed: {e}")

    sparse_pts = []
    try:
        res_sparse = client.query_points(
            collection_name=collection_name,
            query=models.SparseVector(
                indices=sparse_emb.indices.tolist(),
                values=sparse_emb.values.tolist(),
            ),
            using="sparse",
            query_filter=q_filter,
            limit=RETRIEVAL_TOP_K,
        )
        sparse_pts = getattr(res_sparse, "points", []) or []
    except Exception as e:
        print(f"[WARN] Sparse Qdrant search failed: {e}")

    # 3. Fuse & Rerank
    fused_pts = reciprocal_rank_fusion(dense_pts, sparse_pts)
    final_pts = rerank_candidates(query, fused_pts, top_k=top_k)

    # Convert to structured list of dicts
    top_jobs = []
    for rank, pt in enumerate(final_pts, start=1):
        payload = dict(pt.payload or {})
        score = getattr(pt, "score", 0.0)
        job_dict = {"rank": rank, "score": score}
        for k, v in payload.items():
            job_dict[k] = v
        # Ensure all 35 schema keys exist
        for col in ALL_SCHEMA_COLUMNS:
            if col not in job_dict:
                job_dict[col] = payload.get(col)
        top_jobs.append(job_dict)

    return top_jobs


# ============================================================
# LLM Filter Extraction
# ============================================================

ALLOWED_FILTER_KEYS = {
    "jobTitle", "company", "industry", "companySize", "city", "state", "country",
    "remoteHybrid", "wfh", "onsite", "experienceLevel", "minExperienceYears",
    "salaryMin", "salaryMax", "jobCategory", "verifiedStatus"
}


def _extract_filters_via_llm(query: str, model_name: str = DEFAULT_MODEL) -> Dict[str, Any]:
    """Extract structured metadata filters via Ollama LLM matching schema fields."""
    prompt = f"""You are a job search metadata filter extractor.
Extract any relevant metadata filters explicitly mentioned in the user query for querying our jobs database.

Available filter fields:
- company (string): Company name (e.g. "Cognizant", "Google")
- city (string): City (e.g. "Bengaluru", "Pune")
- country (string): Country (e.g. "India", "USA")
- remoteHybrid (boolean): true if Remote, false if Onsite/Hybrid
- wfh (boolean): true or false
- onsite (boolean): true or false
- experienceLevel (string): "Entry Level", "Associate", "Mid Level", "Senior Level", "Lead Level"
- minExperienceYears (integer): Minimum years of experience (e.g. 2, 3)
- salaryMin (integer): Minimum salary (e.g. 80000)
- salaryMax (integer): Maximum salary (e.g. 150000)

Query: "{query}"

Instructions:
1. ONLY include fields that are explicitly mentioned in the query.
2. For numeric fields, return pure numbers (e.g. 50000, not "50k").
3. Return ONLY a valid JSON object. If no filters apply, return {{}}.
"""
    try:
        response = ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.1},
            format="json",
        )
        raw_text = response.get("message", {}).get("content", "{}")
        parsed = safe_parse_json(raw_text)

        extracted: Dict[str, Any] = {}
        for k, v in parsed.items():
            if k in ALLOWED_FILTER_KEYS and v is not None and v != "":
                if k in ("salaryMin", "salaryMax", "minExperienceYears"):
                    c_num = clean_number(v)
                    if c_num is not None:
                        extracted[k] = int(c_num)
                else:
                    extracted[k] = v
        return extracted
    except Exception as e:
        print(f"[WARN] Filter extraction failed: {e}")
        return {}


# ============================================================
# LangGraph Node
# ============================================================

def job_retrieval_and_recommendation_node(state: AgentState) -> Dict[str, Any]:
    """
    Node: Queries Qdrant within the active project's collection, logs the turn,
    and streams comprehensive recommendations covering all 35 schema fields with
    a proactive follow-up offer to auto-apply.
    """
    query = state.get("user_query", "")
    thread_id = state.get("thread_id", "default_session")
    model_name = state.get("model_name", DEFAULT_MODEL)
    resume_text = state.get("resume_text")
    history_summary = state.get("history_summary")
    recent_messages = state.get("recent_messages", [])
    collection_name = state.get("collection_name") or "linkedin_jobs"
    project_name = state.get("project_name")
    cb = state.get("status_callback")

    # 1. Extract metadata filters
    if cb:
        cb("Extracting search filters from query...")
    meta_filters = _extract_filters_via_llm(query, model_name=model_name)

    # 2. Retrieve top 5 jobs from project's Qdrant collection
    if cb:
        cb(f"Retrieving top matching jobs from collection '{collection_name}'...")
    try:
        retrieved_jobs = search_jobs(
            query=query,
            collection_name=collection_name,
            top_k=FINAL_TOP_K,
            **meta_filters,
        )
    except Exception as e:
        print(f"[WARN] Qdrant search error: {e}")
        retrieved_jobs = []

    # 3. Assemble system prompt for presenting all jobs with 35 fields
    n = len(retrieved_jobs)
    system_prompt = (
        "You are an elite LinkedIn Career Strategist and Senior Talent Acquisition Executive.\n"
        "You have complete conversational memory and access to the candidate's resume and job dataset.\n"
        "Always maintain context from previous turns."
    )

    if history_summary:
        system_prompt += f"\n\n### Summary of Earlier Conversation History:\n{history_summary}"

    if resume_text:
        system_prompt += f"\n\n### Candidate Resume Profile:\n{resume_text}"

    if retrieved_jobs:
        system_prompt += f"\n\n### Active {n} Retrieved Jobs from Database:\n{format_jobs_for_llm(retrieved_jobs)}"
        system_prompt += (
            "\n\n--- MANDATORY INSTRUCTIONS FOR PRESENTING RETRIEVED JOBS ---\n"
            f"CRITICAL REQUIREMENT: You have received EXACTLY {n} jobs. You MUST present ALL {n} jobs without omitting any.\n\n"
            f"For EVERY single job (Job #1 to Job #{n}), display all available payload details:\n"
            "- **Job Title**: ...\n"
            "- **Company**: ...\n"
            "- **Industry & Size**: Industry, Company Size\n"
            "- **Location & Work Mode**: City, State, Country, Remote/Hybrid/Onsite\n"
            "- **Experience & Skills**: Experience Level, Experience Years, Skills Required\n"
            "- **Salary**: Salary Min to Salary Max\n"
            "- **Dates**: Date Posted, Date Found, Verified Status\n"
            "- **Company Links & Contacts**: Website, LinkedIn, Company Email, Phone, Founded/HQ\n"
            "- **Hiring Contact Details**: Name, Title, Designation, LinkedIn, Contact Info\n"
            "- **Apply Link**: Source URL\n"
            "- **Job Description & Summary**: Overview of role and responsibilities\n\n"
            f"After presenting complete details for ALL {n} jobs:\n"
            "1. **Full JD vs Resume Comparative Analysis & Final Ranking**:\n"
            "   - Thoroughly compare the candidate's Resume profile against the Full Job Description (`jobDescription`) and responsibilities of each of the 5 jobs.\n"
            "   - Based on this deep JD-to-Resume alignment, provide a definitive RANKING of these jobs (Rank #1 being the absolute best fit, down to Rank #5).\n"
            "   - For each rank, give concrete, specific reasons referencing exact requirements from the Job Description and matching experiences from the resume.\n"
            "2. **Strategic Recommendations**: Action plan on which role(s) to apply to immediately.\n"
            "3. **Skill Gap Analysis**: Specific missing tools, certifications, or technologies for each role.\n"
            "4. **Recruiter Outreach Guide**: Personalized outreach advice for the hiring contacts.\n"
            "5. **Proactive Application Offer**: Conclude with a clear next step prompt:\n"
            "   '💡 **Next Step**: If you would like, I can draft a tailored application email and customized cover letter on your behalf for this top position (or any of the listed roles) and prepare it for immediate submission. Would you like me to prepare an application for Job #1? (Simply say: \"Apply for Job 1\" or \"Apply to this role\")'\n\n"
            "REMEMBER: Display EVERY job accurately without skipping details!"
        )
    else:
        filters_desc = ", ".join([f"{k}: {v}" for k, v in meta_filters.items()]) if meta_filters else "your specified query"
        system_prompt += (
            f"\n\n--- CRITICAL DATABASE NOTICE: ZERO (0) MATCHING JOBS FOUND ---\n"
            f"No jobs in the database collection matched the user's query: '{query}' (Filters: {filters_desc}).\n\n"
            "MANDATORY INSTRUCTIONS FOR THIS TURN:\n"
            "1. Inform the user directly, clearly, and politely that NO jobs matching their exact criteria/role currently exist in this project's database collection.\n"
            "2. STRICT PROHIBITION: DO NOT make up, imagine, hallucinate, or recommend jobs from previous chat turns or from external knowledge. You must be honest that 0 jobs were found in the database for this search.\n"
            "3. Suggest that the user clicks '🔄 Collect New Jobs (Exa Agent)' near the input bar to fetch fresh jobs from Exa.ai, or adjust/broaden their search criteria (such as location or salary range)."
        )

    llm_messages = [{"role": "system", "content": system_prompt}]
    for m in recent_messages:
        role = m.get("role")
        content = m.get("content")
        if role in ["user", "assistant"] and content:
            llm_messages.append({"role": role, "content": str(content)})
    llm_messages.append({"role": "user", "content": query})

    if cb:
        cb("Generating job recommendations via Ollama...")

    stream = ollama.chat(
        model=model_name,
        messages=llm_messages,
        options={"temperature": 0.3, "num_predict": 4096},
        stream=True,
    )

    return {
        "current_jobs": retrieved_jobs,
        "retrieved_jobs": retrieved_jobs,
        "meta_filters": meta_filters,
        "llm_messages": llm_messages,
        "response_stream": stream,
    }
