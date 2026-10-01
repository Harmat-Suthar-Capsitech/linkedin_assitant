"""
CrewAI Custom Tools for LinkedIn AI Career Assistant.
Equips agents with tools for:
1. Qdrant Hybrid Job Retrieval (Dense + Sparse BM25 + RRF + CrossEncoder reranking) with all 35 schema fields.
2. Exa.ai Web Search for real-time company intelligence and career postings.
3. Job Application & Outreach Drafter.
4. Conversation Token Check & Progressive History Summarizer.
"""
from __future__ import annotations

import json
from typing import Dict, Any, List, Optional
from crewai.tools import tool

from nodes.retrieval_node import (
    search_jobs,
    format_jobs_for_llm,
    _extract_filters_via_llm,
    FINAL_TOP_K,
)
from nodes.exa_node import web_search_company
from nodes.apply_node import resolve_target_job_via_llm
# from nodes.history_node import (
#     count_message_tokens,
#     summarize_older_history,
#     MAX_TOKEN_POOL,
#     WELCOME_BANNER,
# )

# Global context for active collection and session state
_active_context: Dict[str, Any] = {
    "collection_name": "linkedin_jobs",
    "cached_jobs": [],
    "resume_text": None,
    "last_retrieved_jobs": [],
    "last_email_draft": None,
    "last_web_results": None,
}


def set_tool_context(
    collection_name: str = "linkedin_jobs",
    cached_jobs: Optional[List[Dict[str, Any]]] = None,
    resume_text: Optional[str] = None,
):
    """Set dynamic runtime context for tools (active collection, cached jobs, resume)."""
    global _active_context
    _active_context["collection_name"] = collection_name or "linkedin_jobs"
    _active_context["cached_jobs"] = cached_jobs or []
    _active_context["resume_text"] = resume_text
    _active_context["last_retrieved_jobs"] = []
    _active_context["last_email_draft"] = None
    _active_context["last_web_results"] = None


def get_tool_context() -> Dict[str, Any]:
    """Retrieve runtime context collected during tool executions."""
    global _active_context
    return _active_context


@tool("Search Project Jobs Database")
def search_project_jobs_tool(query: str) -> str:
    """
    Search the active project's Qdrant vector database for matching job openings.
    Executes hybrid Dense + Sparse BM25 + RRF retrieval with CrossEncoder reranking.
    Returns all retrieved jobs with all 35 schema fields.
    """
    global _active_context
    coll = _active_context.get("collection_name", "linkedin_jobs")
    
    try:
        # Extract filters from query
        meta_filters = _extract_filters_via_llm(query)
        jobs = search_jobs(
            query=query,
            collection_name=coll,
            top_k=FINAL_TOP_K,
            **meta_filters,
        )
        _active_context["last_retrieved_jobs"] = jobs
        if not jobs:
            return f"ZERO (0) jobs found in project collection '{coll}' for query: '{query}'."
        
        return f"Successfully retrieved {len(jobs)} jobs from collection '{coll}':\n\n" + format_jobs_for_llm(jobs)
    except Exception as e:
        return f"Error executing Qdrant job search: {str(e)}"


@tool("Research Company & Market Intelligence via Web")
def search_company_web_tool(query: str) -> str:
    """
    Search the live web using Exa.ai for real-time company background, active hiring status,
    open job roles, tech stack, and career portal links.
    """
    global _active_context
    try:
        search_data = web_search_company(query)
        _active_context["last_web_results"] = search_data
        results = search_data.get("results", [])
        if not results:
            return f"No real-time web results found for '{query}'."
        
        blocks = []
        for idx, r in enumerate(results, start=1):
            title = r.get("title", "Source")
            url = r.get("url", "")
            text = r.get("text", "")
            blocks.append(f"[{idx}] {title} ({url})\n{text}")
        return "\n\n".join(blocks)
    except Exception as e:
        return f"Error executing live web search: {str(e)}"


@tool("Prepare Job Application and Recruiter Outreach")
def prepare_job_application_tool(target_specifier: str) -> str:
    """
    Resolves the target job from cached jobs or company name, identifies the recruiter/hiring email,
    and formats the target details for application and email drafting.
    All target selection and email resolution decisions are performed by LLM.
    """
    global _active_context
    cached_jobs = _active_context.get("cached_jobs", [])
    
    resolved = resolve_target_job_via_llm(
        query=target_specifier,
        cached_jobs=cached_jobs,
    )
    
    target_job = resolved.get("target_job")
    rank_num = resolved.get("rank", 1)
    target_company = resolved.get("company", "Target Company")
    target_title = resolved.get("job_title", "Role")
    to_email = resolved.get("recipient_email", "careers@company.com")
    contact_name = resolved.get("contact_name", "Hiring Team")
    source_url = resolved.get("source_url", "")
    
    draft_info = {
        "to_email": to_email,
        "company": target_company,
        "job_title": target_title,
        "contact_name": contact_name,
        "source_url": source_url,
        "rank": rank_num,
        "target_job": target_job,
    }
    _active_context["last_email_draft"] = draft_info

    if target_job:
        return (
            f"Target Job Resolved: Job #{rank_num} - {target_title} at {target_company}.\n"
            f"Recipient Email: {to_email}\n"
            f"Contact Person: {contact_name}\n"
            f"Source URL: {source_url}\n"
            f"Job Description Excerpt: {str(target_job.get('jobDescription') or target_job.get('jobSummary') or '')[:500]}"
        )
    else:
        return (
            f"Direct Company Application Resolved for: {target_company}.\n"
            f"Discovered Hiring Email: {to_email}\n"
            f"Recipient: {contact_name}"
        )
