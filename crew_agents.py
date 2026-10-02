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

DEFAULT_MODEL = os.getenv("LLM_MODEL", "bjoernb/gemma4-31b-fast:latest")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL") or os.getenv("OLLAMA_HOST") or "http://localhost:11434"
# Sync with ollama python package which relies on OLLAMA_HOST
os.environ["OLLAMA_HOST"] = OLLAMA_BASE_URL


def get_crew_llm(model_name: str = DEFAULT_MODEL) -> LLM:
    """Create or return LiteLLM-compatible Ollama LLM instance for CrewAI."""
    formatted_model = model_name if model_name.startswith("ollama/") else f"ollama/{model_name}"
    return LLM(
        model=formatted_model,
        base_url="https://ollama.com/v1",
        api_key=os.getenv("OLLAMA_API_KEY"),
        temperature=0.2,
    )


# ============================================================
# Specialized CrewAI Agents
# ============================================================

def create_history_agent(llm: Optional[LLM] = None) -> Agent:
    """Conversation Memory & Token Pool Specialist."""
    return Agent(
        role="Conversation Memory & Context Specialist",
        goal="Manage conversational token pools, summarize older chat turns progressively, and preserve candidate background and constraints.",
        backstory=(
            "You are an expert conversation memory architect for executive career advisory. "
            "You ensure that critical candidate profile details, preferences, and discussed job opportunities "
            "are preserved with high fidelity while adhering strictly to conversational token budgets."
        ),
        llm=llm or get_crew_llm(),
        verbose=False,
        allow_delegation=False,
    )


def create_router_agent(llm: Optional[LLM] = None) -> Agent:
    """Career Intake & Workflow Coordinator."""
    return Agent(
        role="Career Intake & Workflow Coordinator",
        goal="Accurately classify user query intent into 'job_search', 'web_search', 'apply_job', or 'general' within the dialogue context.",
        backstory=(
            "You are the chief intake coordinator for a top-tier executive talent advisory firm. "
            "You evaluate incoming candidate requests against past dialogue to determine whether the user wants "
            "to search database jobs, research real-time company intelligence, apply to a job, or have general career advice."
        ),
        llm=llm or get_crew_llm(),
        verbose=False,
        allow_delegation=False,
    )


def create_query_rewriter_agent(llm: Optional[LLM] = None) -> Agent:
    """Specialized Semantic Search Query & Vector Embedding Optimization Agent."""
    return Agent(
        role="Semantic Search Query & Vector Embedding Optimization Specialist",
        goal="Analyze user search requests and candidate resumes to synthesize targeted, high-density queries that maximize cosine similarity in Qdrant dense vector search.",
        backstory=(
            "You are an elite Information Retrieval and Dense Vector Search Architect specializing in recruitment matching. "
            "You know that conversational queries (such as 'suggest jobs according to this resume') perform poorly against dense vector representations of job postings because actual job postings do not contain conversational request phrases."
            "Your strength is extracting the primary target job title, core tech stack, seniority, domain keywords, and key skills "
            "from user queries and candidate resumes to engineer a high-density, vector-optimized search query for Qdrant retrieval."
        ),
        llm=llm or get_crew_llm(),
        verbose=False,
        allow_delegation=False,
    )


def create_job_retrieval_agent(llm: Optional[LLM] = None) -> Agent:
    """Executive Talent Acquisition & Job Matching Strategist."""
    return Agent(
        role="Senior Executive Talent Strategist & Job Matching Specialist",
        goal="Retrieve jobs from the project database, present all 35 schema fields with complete transparency, perform JD vs Resume comparative gap analysis, rank roles, and offer proactive application next steps.",
        backstory=(
            "You are a seasoned Head of Executive Talent Acquisition. "
            "When presenting job opportunities, you never skip any available payload fields—you show complete 35-field dossiers. "
            "You critically compare full job descriptions against candidate resumes, giving honest rankings and actionable recruiter outreach strategies."
        ),
        tools=[search_project_jobs_tool],
        llm=llm or get_crew_llm(),
        verbose=False,
        allow_delegation=False,
    )


def create_company_intelligence_agent(llm: Optional[LLM] = None) -> Agent:
    """Real-Time Talent Market & Company Intelligence Researcher."""
    return Agent(
        role="Talent Market & Company Intelligence Researcher",
        goal="Conduct live web research using Exa.ai on companies, active hiring status, open roles, culture, and career portal links.",
        backstory=(
            "You are an elite corporate intelligence analyst. You track real-time hiring trends, company business developments, "
            "workplace culture, and open role availability across global organizations using real-time search."
        ),
        tools=[search_company_web_tool],
        llm=llm or get_crew_llm(),
        verbose=False,
        allow_delegation=False,
    )


def create_job_application_agent(llm: Optional[LLM] = None) -> Agent:
    """Executive Job Application & Recruiter Outreach Specialist."""
    return Agent(
        role="Executive Job Application & Recruiter Outreach Specialist",
        goal="Resolve target job and hiring email contacts, craft personalized, high-converting application emails and cover letters from the candidate's resume, and prepare email sending payloads.",
        backstory=(
            "You are a master of high-impact executive recruiter outreach. You craft tailored application emails and cover letters "
            "that connect candidate achievements directly to job requirements, ensuring maximum callback rates."
        ),
        tools=[prepare_job_application_tool],
        llm=llm or get_crew_llm(),
        verbose=False,
        allow_delegation=False,
    )


def create_general_career_agent(llm: Optional[LLM] = None) -> Agent:
    """Senior LinkedIn Career Advisor & Follow-Up Counselor."""
    return Agent(
        role="Senior LinkedIn Career Advisor & Follow-Up Counselor",
        goal="Deliver tailored career guidance, resume suggestions, interview strategies, and comparative evaluations of previously retrieved jobs.",
        backstory=(
            "You are an empathetic, world-class career mentor and coach. You help candidates navigate career decisions, "
            "evaluate competing job offers, identify upskilling opportunities, and answer follow-up queries with high precision."
        ),
        llm=llm or get_crew_llm(),
        verbose=False,
        allow_delegation=False,
    )
