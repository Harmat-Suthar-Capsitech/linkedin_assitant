"""
Module: Job Application Target Resolution Tool & CrewAI Agent Integration.
Used by prepare_job_application_tool in CrewAI Multi-Agent workflow.
"""
from __future__ import annotations

from typing import Dict, Any, List, Optional
import ollama

from logger_util import safe_parse_json

DEFAULT_MODEL = "bjoernb/gemma4-31b-fast:latest"


def resolve_target_job_via_llm(
    query: str,
    cached_jobs: Optional[List[Dict[str, Any]]] = None,
    model_name: str = DEFAULT_MODEL,
) -> Dict[str, Any]:
    """
    Helper tool function used by prepare_job_application_tool in CrewAI
    to parse target company, role, recruiter email, and contact person.
    """
    cached_jobs = cached_jobs or []

    if not cached_jobs:
        prompt = f"""You are an executive talent acquisition specialist.
The user wants to apply to a specific company or role directly.

User Inquiry: "{query}"

Analyze the request and determine:
1. Target company name.
2. Target job title (or 'Open Technical Opportunities' if unspecified).
3. The official recruitment or careers email address (e.g., careers@company.com).
4. The hiring contact or department name (e.g., 'Talent Acquisition Team').

Return ONLY a valid JSON object:
{{
  "selected_rank": null,
  "company": "<company name>",
  "job_title": "<job title>",
  "recipient_email": "<careers email address>",
  "contact_name": "<contact person or department>"
}}
"""
    else:
        job_lines = []
        for j in cached_jobs:
            rank = j.get("rank", 1)
            comp = j.get("company") or j.get("Company") or "Company"
            title = j.get("jobTitle") or j.get("Job Title") or "Role"
            email = j.get("companyEmail") or j.get("Company Email") or j.get("hiringContactInfo") or "Not provided"
            contact = j.get("hiringContactName") or j.get("Hiring Contact Name") or "Hiring Team"
            url = j.get("sourceUrl") or j.get("Source URL") or ""
            job_lines.append(
                f"- Job #{rank}: Title: '{title}' | Company: '{comp}' | Email: '{email}' | Contact: '{contact}' | URL: '{url}'"
            )
        jobs_summary = "\n".join(job_lines)

        prompt = f"""You are an executive talent acquisition specialist.
The user wants to apply to one of the active retrieved jobs.

User Inquiry: "{query}"

Active Retrieved Jobs in Session:
{jobs_summary}

Analyze the user's intent and decide:
1. Which job number (rank) the user is targeting. If ambiguous or not specified, choose Job #1.
2. Target company name and role title.
3. Best recipient email address (use email from the job payload if valid, otherwise derive the official careers email for that company).
4. Hiring contact or recruiter name.

Return ONLY a valid JSON object:
{{
  "selected_rank": <integer rank, e.g. 1>,
  "company": "<company name>",
  "job_title": "<job title>",
  "recipient_email": "<careers email address>",
  "contact_name": "<contact person or department>"
}}
"""

    try:
        response = ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.1},
        )
        content = response.get("message", {}).get("content", "{}")
        data = safe_parse_json(content)

        rank = data.get("selected_rank")
        target_job = None
        if isinstance(rank, int) and cached_jobs:
            for j in cached_jobs:
                if j.get("rank") == rank:
                    target_job = j
                    break
            if not target_job and 1 <= rank <= len(cached_jobs):
                target_job = cached_jobs[rank - 1]

        if not target_job and cached_jobs:
            for j in cached_jobs:
                comp = str(j.get("company") or "").lower()
                if comp and comp in str(data.get("company", "")).lower():
                    target_job = j
                    rank = j.get("rank", 1)
                    break
            if not target_job:
                target_job = cached_jobs[0]
                rank = target_job.get("rank", 1)

        company = data.get("company") or (target_job.get("company") if target_job else "Target Company")
        job_title = data.get("job_title") or (target_job.get("jobTitle") if target_job else "Role")
        recipient_email = data.get("recipient_email")
        if not recipient_email or "@" not in str(recipient_email):
            clean_co = company.lower().replace(" ", "").replace(".", "")
            recipient_email = f"careers@{clean_co or 'company'}.com"
        contact_name = data.get("contact_name") or (target_job.get("hiringContactName") if target_job else "Hiring Team")

        return {
            "target_job": target_job,
            "rank": rank or 1,
            "company": company,
            "job_title": job_title,
            "recipient_email": recipient_email,
            "contact_name": contact_name,
            "source_url": target_job.get("sourceUrl", "") if target_job else "",
        }
    except Exception as e:
        print(f"[WARN] Target job resolution failed: {e}")
        target_job = cached_jobs[0] if cached_jobs else None
        return {
            "target_job": target_job,
            "rank": 1,
            "company": target_job.get("company", "Company") if target_job else "Company",
            "job_title": target_job.get("jobTitle", "Role") if target_job else "Role",
            "recipient_email": "careers@company.com",
            "contact_name": "Hiring Team",
            "source_url": target_job.get("sourceUrl", "") if target_job else "",
        }


def apply_job_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    CrewAI Job Application Specialist Agent node.
    Resolves target contacts and drafts tailored cover letters via autonomous agent tasks.
    """
    from crew_workflow import run_job_application_crew

    query = state.get("user_query", "")
    cached_jobs = state.get("current_jobs", [])
    resume_text = state.get("resume_text")
    collection_name = state.get("collection_name", "linkedin_jobs")
    model_name = state.get("model_name", DEFAULT_MODEL)

    return run_job_application_crew(
        user_query=query,
        resume_text=resume_text,
        cached_jobs=cached_jobs,
        collection_name=collection_name,
        model_name=model_name,
    )
