"""
CrewAI Multi-Agent System for LinkedIn AI Career Assistant.
Defines specialized autonomous agents with distinct roles, goals, backstories,
assigned tools, and inter-agent communication channels.
"""
from __future__ import annotations

import os
from typing import Optional
from crewai import Agent, LLM

from crew_tools import (
    search_project_jobs_tool,
    search_company_web_tool,
    prepare_job_application_tool,
)

DEFAULT_MODEL = os.getenv("LLM_MODEL", "gemma4:31b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL") or os.getenv("OLLAMA_HOST") or "https://ollama.com/v1"
# Sync with ollama python package which relies on OLLAMA_HOST
os.environ["OLLAMA_HOST"] = OLLAMA_BASE_URL


def get_crew_llm(model_name: str = DEFAULT_MODEL, temperature: float = 0.2) -> LLM:
    """Create LLM instance for CrewAI with tailored temperature.

    Explicitly sets provider='ollama' so CrewAI routes to its native Ollama SDK
    instead of falling through to LiteLLM (which isn't installed on Streamlit Cloud).
    """
    base_url = os.getenv("OLLAMA_BASE_URL") or os.getenv("OLLAMA_HOST") or "https://ollama.com/v1"
    return LLM(
        model=model_name,
        custom_openai=True,
        base_url=base_url,
        api_key=os.getenv("OLLAMA_API_KEY"),
        temperature=temperature,
    )


# ============================================================
# Specialized CrewAI Agents with Tailored Temperatures
# ============================================================

def create_history_agent(
    llm: Optional[LLM] = None,
    model_name: str = DEFAULT_MODEL,
    temperature: float = 0.1,
) -> Agent:
    """Conversation Memory & Token Pool Specialist (Low temperature 0.1 for high factual retention)."""
    return Agent(
        role="Conversation Memory & Context Specialist",
        goal="Manage conversational token pools, summarize older chat turns progressively, and preserve candidate background and constraints.",
        backstory=(
            "You are an expert conversation memory architect for executive career advisory. "
            "You ensure that critical candidate profile details, preferences, and discussed job opportunities "
            "are preserved with high fidelity while adhering strictly to conversational token budgets."
        ),
        llm=llm or get_crew_llm(model_name=model_name, temperature=temperature),
        verbose=False,
        allow_delegation=False,
    )


def create_router_agent(
    llm: Optional[LLM] = None,
    model_name: str = DEFAULT_MODEL,
    temperature: float = 0.0,
) -> Agent:
    """Career Intake & Workflow Coordinator (Deterministic temperature 0.0 for consistent classification)."""
    return Agent(
        role="Career Intake & Workflow Coordinator",
        goal="Accurately classify user query intent into 'job_search', 'web_search', 'apply_job', or 'general' within the dialogue context.",
        backstory=(
            "You are the chief intake coordinator for a top-tier executive talent advisory firm. "
            "You evaluate incoming candidate requests against past dialogue to determine whether the user wants "
            "to search database jobs, research real-time company intelligence, apply to a job, or have general career advice."
        ),
        llm=llm or get_crew_llm(model_name=model_name, temperature=temperature),
        verbose=False,
        allow_delegation=False,
    )


def create_query_rewriter_agent(
    llm: Optional[LLM] = None,
    model_name: str = DEFAULT_MODEL,
    temperature: float = 0.1,
) -> Agent:
    """Semantic Search Query & Vector Embedding Optimization Agent (Low temperature 0.1 for precise keyword extraction)."""
    return Agent(
        role="Semantic Search Query & Vector Embedding Optimization Specialist",
        goal="Analyze user search requests and candidate resumes to synthesize targeted, high-density queries that maximize cosine similarity in Qdrant dense vector search.",
        backstory=(
            "You are an elite Information Retrieval and Dense Vector Search Architect specializing in recruitment matching. "
            "You know that conversational queries perform poorly against dense vector representations of job postings. "
            "Your strength is extracting the primary target job title, core tech stack, seniority, domain keywords, and key skills "
            "from user queries and candidate resumes to engineer a high-density, vector-optimized search query for Qdrant retrieval."
        ),
        llm=llm or get_crew_llm(model_name=model_name, temperature=temperature),
        verbose=False,
        allow_delegation=False,
    )


def create_job_retrieval_agent(
    llm: Optional[LLM] = None,
    model_name: str = DEFAULT_MODEL,
    temperature: float = 0.2,
) -> Agent:
    """Executive Talent Acquisition & Job Matching Strategist (Temperature 0.2 for strict 35-field schema data handling)."""
    return Agent(
        role="Senior Executive Talent Strategist & Job Matching Specialist",
        goal="Retrieve jobs from the project database, present all 35 schema fields with complete transparency, perform JD vs Resume comparative gap analysis, rank roles, and offer proactive application next steps.",
        backstory=(
            "You are a seasoned Head of Executive Talent Acquisition. "
            "When presenting job opportunities, you never skip any available payload fields—you show complete 35-field dossiers. "
            "You critically compare full job descriptions against candidate resumes, giving honest rankings and actionable recruiter outreach strategies."
        ),
        tools=[search_project_jobs_tool],
        llm=llm or get_crew_llm(model_name=model_name, temperature=temperature),
        verbose=False,
        allow_delegation=False,
    )


def create_company_intelligence_agent(
    llm: Optional[LLM] = None,
    model_name: str = DEFAULT_MODEL,
    temperature: float = 0.2,
) -> Agent:
    """Real-Time Talent Market & Company Intelligence Researcher (Temperature 0.2 for factual web research synthesis)."""
    return Agent(
        role="Talent Market & Company Intelligence Researcher",
        goal="Conduct live web research using Exa.ai on companies, active hiring status, open roles, culture, and career portal links.",
        backstory=(
            "You are an elite corporate intelligence analyst. You track real-time hiring trends, company business developments, "
            "workplace culture, and open role availability across global organizations using real-time search."
        ),
        tools=[search_company_web_tool],
        llm=llm or get_crew_llm(model_name=model_name, temperature=temperature),
        verbose=False,
        allow_delegation=False,
    )


def create_job_application_agent(
    llm: Optional[LLM] = None,
    model_name: str = DEFAULT_MODEL,
    temperature: float = 0.5,
) -> Agent:
    """Executive Job Application & Recruiter Outreach Specialist (Temperature 0.5 for persuasive, creative application drafting)."""
    return Agent(
        role="Executive Job Application & Recruiter Outreach Specialist",
        goal="Resolve target job and hiring email contacts, craft personalized, high-converting application emails and cover letters from the candidate's resume, and prepare email sending payloads.",
        backstory=(
            "You are a master of high-impact executive recruiter outreach. You craft tailored application emails and cover letters "
            "that connect candidate achievements directly to job requirements, ensuring maximum callback rates."
        ),
        tools=[prepare_job_application_tool],
        llm=llm or get_crew_llm(model_name=model_name, temperature=temperature),
        verbose=False,
        allow_delegation=False,
    )


def create_general_career_agent(
    llm: Optional[LLM] = None,
    model_name: str = DEFAULT_MODEL,
    temperature: float = 0.4,
) -> Agent:
    """Senior LinkedIn Career Advisor & Follow-Up Counselor (Temperature 0.4 for engaging, empathetic career mentoring)."""
    return Agent(
        role="Senior LinkedIn Career Advisor & Follow-Up Counselor",
        goal="Deliver tailored career guidance, resume suggestions, interview strategies, and comparative evaluations of previously retrieved jobs.",
        backstory=(
            "You are an empathetic, world-class career mentor and coach. You help candidates navigate career decisions, "
            "evaluate competing job offers, identify upskilling opportunities, and answer follow-up queries with high precision."
        ),
        llm=llm or get_crew_llm(model_name=model_name, temperature=temperature),
        verbose=False,
        allow_delegation=False,
    )
