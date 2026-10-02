"""
CrewAI Multi-Agent Workflow Engine for LinkedIn AI Career Assistant.
Orchestrates autonomous agents collaborating across tasks using CrewAI Flows and Crews:
1. CareerAssistantFlow (CrewAI Flow):
   - State management via Pydantic CareerFlowState
   - History & Memory Agent: Token budget management and progressive summarization.
   - Intake & Routing Coordinator Agent: Intent classification and dynamic workflow routing.
   - Multi-Agent Specialist Crews:
     * Job Search Crew: Collaborative Job Retrieval Strategist + Career Evaluation Agent (35 schema fields).
     * Company Intelligence Crew: Real-Time Market & Corporate Intelligence Researcher (Exa.ai).
     * Application & Outreach Crew: Target Job Resolution + Recruiter Email Drafter.
     * General Career Advisory Crew: Senior LinkedIn Career Mentor & Evaluation Counselor.
"""
from __future__ import annotations

import os
import re
from typing import Dict, Any, List, Optional, Generator, Callable
from pydantic import BaseModel, Field

from crewai import Agent, Task, Crew, Process, LLM
from crewai.flow.flow import Flow, start, listen, router

from crew_agents import (
    get_crew_llm,
    create_history_agent,
    create_router_agent,
    create_query_rewriter_agent,
    create_job_retrieval_agent,
    create_company_intelligence_agent,
    create_job_application_agent,
    create_general_career_agent,
    DEFAULT_MODEL,
)
from crew_tools import (
    set_tool_context,
    get_tool_context,
    search_project_jobs_tool,
    search_company_web_tool,
    prepare_job_application_tool,
)
from nodes.history_node import (
    count_message_tokens,
    summarize_older_history,
    MAX_TOKEN_POOL,
    WELCOME_BANNER,
)
from logger_util import safe_parse_json


# ============================================================
# Flow State
# ============================================================

class CareerFlowState(BaseModel):
    user_query: str = ""
    search_query: str = ""
    thread_id: str = "default_session"
    resume_text: Optional[str] = None
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    history_summary: Optional[str] = None
    history_context: str = ""
    cached_jobs: List[Dict[str, Any]] = Field(default_factory=list)
    retrieved_jobs: Optional[List[Dict[str, Any]]] = None
    email_draft: Optional[Dict[str, Any]] = None
    meta_filters: Dict[str, Any] = Field(default_factory=dict)
    intent: str = "general"
    final_response: str = ""
    project_id: Optional[str] = None
    project_name: Optional[str] = None
    collection_name: str = "linkedin_jobs"
    conversation_title: str = "New Chat"
    model_name: str = DEFAULT_MODEL


# ============================================================
# Standalone CrewAI Agent Runners
# ============================================================

def classify_intent_with_router_agent(
    user_query: str,
    history_context: str = "",
    has_cached_jobs: bool = False,
    model_name: str = DEFAULT_MODEL,
) -> str:
    """
    Intake & Routing Coordinator Agent analyzes the user query
    in the context of conversation history and cached jobs using CrewAI.
    Deterministic temperature (0.0) ensures accurate intent classification.
    """
    router_agent = create_router_agent(model_name=model_name, temperature=0.0)

    routing_task = Task(
        description=f"""Analyze the user's query within the provided conversation history and memory.
Conversation History:
{history_context or "(No prior dialogue)"}

Cached Jobs Currently in Memory: {'Yes' if has_cached_jobs else 'No'}
Latest User Query: "{user_query}"

Classify into EXACTLY ONE category:
1. "job_search" -> When user is searching or asking for jobs/vacancies, or requesting job recommendations based on their resume/profile (e.g. "Find python developer jobs", "Looking for backend roles in Pune", "suggest jobs according to this resume", "find jobs matching my resume").
2. "web_search" -> When user is asking for real-time company intelligence, hiring status, or culture (e.g. "Is Google hiring for AI engineers in India?", "Tell me about Attis company").
3. "apply_job" -> When user asks to apply to a specific job or company, draft an email, or agrees to apply (e.g. "Apply to Job 1", "Apply to Job 2", "Apply to Microsoft").
4. "general" -> Greetings, introductions, general career advice, or follow-up questions comparing currently cached jobs (e.g. "suggest best one job from these 5 jobs", "compare job 1 and 3").

Return ONLY a JSON object:
{{"intent": "job_search" | "web_search" | "apply_job" | "general"}}
""",
        expected_output='A JSON object with key "intent"',
        agent=router_agent,
    )

    crew = Crew(agents=[router_agent], tasks=[routing_task], verbose=False)
    output = crew.kickoff()
    raw = str(output.raw if hasattr(output, "raw") else output).strip()

    parsed = safe_parse_json(raw)
    intent = parsed.get("intent", "").strip().lower()
    if intent not in ("job_search", "web_search", "apply_job"):
        intent = "general"

    return intent


def run_company_intelligence_crew(
    user_query: str,
    history_summary: Optional[str] = None,
    model_name: str = DEFAULT_MODEL,
) -> str:
    """Uses the CrewAI Company Intelligence Researcher Agent with factual temperature (0.2) and search_company_web_tool."""
    intel_agent = create_company_intelligence_agent(model_name=model_name, temperature=0.2)

    research_task = Task(
        description=f"""Conduct live web research using search_company_web_tool for the user's inquiry:
Inquiry: "{user_query}"

Earlier Context:
{history_summary if history_summary else "(None)"}

INSTRUCTIONS:
- Research company background, active hiring status, open positions, culture, and career portal links.
- Synthesize the findings into clear, well-structured bullet points.
- Cite source URLs and specify current hiring status.
""",
        expected_output="A detailed company intelligence synthesis with citations and career links.",
        agent=intel_agent,
        tools=[search_company_web_tool],
    )

    crew = Crew(agents=[intel_agent], tasks=[research_task], verbose=False)
    result = crew.kickoff()
    return str(result.raw if hasattr(result, "raw") else result)


def run_job_application_crew(
    user_query: str,
    resume_text: Optional[str] = None,
    cached_jobs: Optional[List[Dict[str, Any]]] = None,
    collection_name: str = "linkedin_jobs",
    model_name: str = DEFAULT_MODEL,
) -> Dict[str, Any]:
    """Uses the CrewAI Job Application & Recruiter Outreach Specialist Agent with persuasive temperature (0.5)."""
    set_tool_context(
        collection_name=collection_name,
        cached_jobs=cached_jobs or [],
        resume_text=resume_text,
    )
    app_agent = create_job_application_agent(model_name=model_name, temperature=0.5)

    prep_task = Task(
        description=f"""Identify the target job or company the user wants to apply to from their query:
Query: "{user_query}"

Use prepare_job_application_tool to extract target job details, recipient email, and hiring contact.
""",
        expected_output="Target job details including recipient email, hiring contact, and role summary.",
        agent=app_agent,
        tools=[prepare_job_application_tool],
    )

    draft_task = Task(
        description=f"""Based on the target job details from prepare_job_application_tool and the candidate's resume,
craft a high-converting personalized job application email and cover letter.

Candidate Resume Profile:
{resume_text if resume_text else "(Candidate has relevant technical background)"}

INSTRUCTIONS:
1. Subject line: Write a compelling, professional subject line.
2. Salutation: Address the hiring contact or company talent team respectfully.
3. Tailored 3-4 paragraph Cover Letter:
   - Para 1: Enthusiasm for the role and company.
   - Para 2: Specific matching skills and achievements from resume.
   - Para 3: Value proposition and mention that resume is attached.
   - Para 4: Call to action for an interview.
4. Present the complete email draft and options to send via App Mail or personal Gmail.
""",
        expected_output="Personalized cold outreach email, subject line, and cover letter formatted for display.",
        agent=app_agent,
        context=[prep_task],
    )

    crew = Crew(
        agents=[app_agent],
        tasks=[prep_task, draft_task],
        process=Process.sequential,
        verbose=False,
    )

    result = crew.kickoff()
    response_text = str(result.raw if hasattr(result, "raw") else result)

    tool_ctx = get_tool_context()
    draft = tool_ctx.get("last_email_draft")
    email_draft_payload = None
    if draft:
        subj_match = re.search(r'\*\*Subject\*\*:\s*`?([^\n`]+)`?', response_text, re.IGNORECASE)
        subject = subj_match.group(1).strip() if subj_match else f"Application for {draft.get('job_title', 'Role')} - Candidate"
        email_draft_payload = {
            "job_rank": draft.get("rank", 1),
            "job_title": draft.get("job_title", "Role"),
            "company": draft.get("company", "Company"),
            "location": draft.get("target_job", {}).get("city", "N/A") if draft.get("target_job") else "N/A",
            "recipient_email": draft.get("to_email", "careers@company.com"),
            "recipient_name": draft.get("contact_name", "Hiring Team"),
            "subject": subject,
            "body": response_text,
            "status": "ready_to_send",
        }

    # Append interactive send-option card if not present
    if "### ✉️ Ready to Send!" not in response_text:
        response_text += (
            "\n\n---\n\n"
            "### ✉️ Ready to Send!\n"
            "Please choose how you would like to send this application from the **'Send Application Email'** section below:\n\n"
            "1. **Option 1 (App Mail)**: Send via `apply@ourapp.com` with your personal email in `Reply-To`.\n"
            "2. **Option 2 (Your Gmail)**: Send directly from your personal Gmail using a 16-digit Google App Password.\n"
            "📎 *Your uploaded resume will be attached to the email!*"
        )

    return {
        "email_draft": email_draft_payload,
        "final_text": response_text,
        "target_job": draft.get("target_job") if draft else None,
        "response_stream": [{"message": {"content": response_text}}],
    }


def run_general_career_crew(
    user_query: str,
    resume_text: Optional[str] = None,
    cached_jobs: Optional[List[Dict[str, Any]]] = None,
    history_context: str = "",
    model_name: str = DEFAULT_MODEL,
) -> str:
    """Uses the CrewAI Senior Career Advisor Agent with empathetic temperature (0.4) for mentoring, resume guidance, or comparison."""
    advisor_agent = create_general_career_agent(model_name=model_name, temperature=0.4)

    active_jobs_context = ""
    if cached_jobs:
        from nodes.retrieval_node import format_jobs_for_llm
        active_jobs_context = f"\n\nActive {len(cached_jobs)} Retrieved Jobs in Memory:\n{format_jobs_for_llm(cached_jobs)}"

    advisory_task = Task(
        description=f"""Address the user's inquiry with high-impact career guidance, executive coaching, or job comparison.
User Inquiry: "{user_query}"

Earlier Context & Memory:
{history_context}

Candidate Resume Profile:
{resume_text if resume_text else "(No resume provided)"}
{active_jobs_context}

INSTRUCTIONS:
- If the user introduces themselves, acknowledge warmly and remember their background.
- If asking to compare, rank, or select from active jobs, reference specific titles, companies, and requirements from the active jobs payload.
- Give concrete, actionable advice. Do NOT invent fake jobs.
""",
        expected_output="Warm, actionable career advice and evaluation formatted in markdown.",
        agent=advisor_agent,
    )

    crew = Crew(agents=[advisor_agent], tasks=[advisory_task], verbose=False)
    result = crew.kickoff()
    return str(result.raw if hasattr(result, "raw") else result)


# ============================================================
# CrewAI Career Assistant Flow
# ============================================================

class CareerAssistantFlow(Flow[CareerFlowState]):
    """
    CrewAI Flow orchestrating the entire lifecycle of a career assistant turn.
    Coordinates specialized agents through dedicated Crews and inter-task context chaining.
    """

    def __init__(self, status_callback: Optional[Callable[[str], None]] = None, **kwargs):
        super().__init__(**kwargs)
        self.status_cb = status_callback

    def notify(self, message: str):
        if self.status_cb:
            try:
                self.status_cb(message)
            except Exception:
                pass

    # ------------------------------------------------------------
    # Step 1: Memory & History Assessment
    # ------------------------------------------------------------
    @start()
    def analyze_history_and_memory(self):
        """
        History Agent examines conversational token budget (150k tokens)
        and performs progressive summarization if needed.
        """
        self.notify("🧠 Conversation Memory Agent: Checking dialogue token budget...")
        filtered = [m for m in self.state.messages if WELCOME_BANNER not in str(m.get("content", ""))]
        total_tokens = count_message_tokens(filtered)

        current_summary = self.state.history_summary or ""
        remaining = list(filtered)

        if total_tokens > MAX_TOKEN_POOL:
            self.notify("🧠 Conversation Memory Agent: Summarizing older conversation turns...")
            target_pool = int(MAX_TOKEN_POOL * 0.8)
            while count_message_tokens(remaining) > target_pool and len(remaining) > 2:
                turn = []
                if remaining and remaining[0].get("role") == "user":
                    turn.append(remaining.pop(0))
                    if remaining and remaining[0].get("role") == "assistant":
                        turn.append(remaining.pop(0))
                else:
                    turn.append(remaining.pop(0))

                current_summary = summarize_older_history(
                    older_messages=turn,
                    existing_summary=current_summary,
                    model_name=self.state.model_name,
                )

        self.state.history_summary = current_summary

        # Format context for inter-agent communication
        context_parts = []
        if current_summary:
            context_parts.append(f"Summary of Earlier Conversation:\n{current_summary}")

        if remaining:
            recent_turns = []
            for m in remaining[-8:]:
                role = m.get("role", "user").capitalize()
                c = str(m.get("content", "")).strip()
                if c:
                    recent_turns.append(f"{role}: {c[:250]}")
            if recent_turns:
                context_parts.append("Recent Dialogue History:\n" + "\n".join(recent_turns))

        self.state.history_context = "\n\n".join(context_parts) if context_parts else "(No prior dialogue)"
        return self.state.history_context

    # ------------------------------------------------------------
    # Step 2: Intake & Routing Coordinator Agent
    # ------------------------------------------------------------
    @listen(analyze_history_and_memory)
    def classify_intent_coordinator(self, history_context: str):
        """
        Intake & Routing Coordinator Agent analyzes the user query
        in the context of conversation history and cached jobs.
        """
        self.notify("🎯 Intake Coordinator Agent: Classifying intent and planning agent workflow...")
        intent = classify_intent_with_router_agent(
            user_query=self.state.user_query,
            history_context=history_context,
            has_cached_jobs=bool(self.state.cached_jobs),
            model_name=self.state.model_name,
        )
        self.state.intent = intent
        return intent

    # ------------------------------------------------------------
    # Step 3: Conditional Branching
    # ------------------------------------------------------------
    @router(classify_intent_coordinator)
    def route_by_intent(self, intent: str):
        if intent == "job_search":
            return "execute_job_search"
        elif intent == "web_search":
            return "execute_web_search"
        elif intent == "apply_job":
            return "execute_job_application"
        else:
            return "execute_general_advisory"

    # ------------------------------------------------------------
    # Branch A: Job Search & Matching Crew
    # ------------------------------------------------------------
    @listen("execute_job_search")
    def handle_job_search_crew(self):
        """
        Collaborative multi-agent execution:
        - Query Rewriter Agent (temp 0.1) optimizes query for dense vector similarity.
        - Job Retrieval Strategist (temp 0.2) searches Qdrant with all 35 schema fields.
        - Career Evaluation Specialist (temp 0.4) compares against resume and ranks.
        """
        # Step A: Query Rewriting Agent (Dense Vector Optimization)
        self.notify("🔄 Query Rewriter Agent: Analyzing query & resume for vector embedding optimization...")
        rewriter_agent = create_query_rewriter_agent(model_name=self.state.model_name, temperature=0.1)

        rewriting_task = Task(
            description=f"""You are an elite Semantic Search & Vector Retrieval Query Specialist.
Your job is to rewrite and optimize the user's search query specifically for DENSE VECTOR SIMILARITY and BM25 sparse matching against job postings in a Qdrant vector database.

Original User Query:
"{self.state.user_query}"

Candidate Resume Profile:
{self.state.resume_text if self.state.resume_text else "(No resume provided)"}

Dialogue History / Context:
{self.state.history_context}

CRITICAL RULES FOR QUERY REWRITING:
1. RESUME-BASED QUERIES:
   - If the user query is generic or asks to find/suggest jobs according to/based on their resume (e.g. "suggest jobs according to this resume", "find jobs matching my profile", "recommend jobs for me"):
     DO NOT use conversational words like "find", "suggest", "according", "resume" because matching their vector embeddings will give poor similarity against actual job posts!
     INSTEAD: Thoroughly analyze the Candidate Resume Profile. Extract the primary target job title (e.g., "Full Stack Python Developer", "Data Scientist", "DevOps Engineer"), top 5-7 core technical skills/technologies (e.g., "FastAPI Python Docker AWS PostgreSQL Redis"), domain, and seniority. Synthesize a clean, high-density query combining the target role and top skills.

2. SPECIFIC ROLE QUERIES:
   - If the user specifies a particular role or skills (e.g., "remote react developer jobs", "looking for ML engineer in Bangalore"):
     Preserve the user's target role, location, and seniority. Enhance it with the most relevant technical keywords and industry synonyms (and if candidate resume has matching strong skills in that domain, integrate them to improve similarity).

3. DENSE VECTOR OPTIMIZATION:
   - Remove all conversational fluff, filler words, question phrasing, greetings, and stop words (e.g. "please find me", "can you show", "I want", "according to my resume").
   - Focus on: [Job Title / Role] + [Key Tech Stack / Skills] + [Domain / Seniority] + [Location / Remote if specified].
   - Output ONLY the rewritten search query. No explanations, no markdown formatting, no quotes, no preamble.
""",
            expected_output="A single concise, high-density search query string containing only target role, key skills, and domain keywords.",
            agent=rewriter_agent,
        )

        rewriter_crew = Crew(agents=[rewriter_agent], tasks=[rewriting_task], verbose=False)
        rewriter_result = rewriter_crew.kickoff()
        raw_rewritten = str(rewriter_result.raw if hasattr(rewriter_result, "raw") else rewriter_result).strip()
        raw_rewritten = raw_rewritten.strip('"').strip("'").strip("`")
        if ":" in raw_rewritten and len(raw_rewritten.split(":", 1)[0]) < 25:
            raw_rewritten = raw_rewritten.split(":", 1)[1].strip()
        rewritten_query = raw_rewritten if raw_rewritten else self.state.user_query
        self.state.search_query = rewritten_query

        self.notify(f"🎯 Query Rewriter Agent: Optimized vector query -> '{rewritten_query}'")
        self.notify(f"🔍 Talent Strategist Agent: Searching Qdrant collection '{self.state.collection_name}' with optimized query...")

        job_agent = create_job_retrieval_agent(model_name=self.state.model_name, temperature=0.2)
        advisor_agent = create_general_career_agent(model_name=self.state.model_name, temperature=0.4)

        # Task 1: Job Retrieval (using optimized vector search query)
        retrieval_task = Task(
            description=f"""Search the project's Qdrant vector database for matching jobs using search_project_jobs_tool.
Optimized Vector Search Query: "{rewritten_query}"
Original User Query: "{self.state.user_query}"
Target Collection: "{self.state.collection_name}"

Use the Search Project Jobs Database tool with query="{rewritten_query}" to retrieve matching jobs.
Output all the retrieved jobs and their 35 payload fields.
""",
            expected_output="Retrieved job records with all 35 schema fields, or clear notice if 0 jobs found.",
            agent=job_agent,
            tools=[search_project_jobs_tool],
        )

        # Task 2: Comparative Matching & Strategic Recommendations
        matching_task = Task(
            description=f"""Based on the job records retrieved by the Talent Strategist and the candidate's resume,
prepare a comprehensive career dossier for the user.

Original User Query: "{self.state.user_query}"
Rewritten Vector Query: "{rewritten_query}"
Candidate Resume Profile:
{self.state.resume_text if self.state.resume_text else "(No resume provided. Provide general evaluation based on industry benchmarks.)"}

Earlier Conversation Summary:
{self.state.history_summary if self.state.history_summary else "(No prior summary)"}

INSTRUCTIONS:
1. If jobs were found:
   - Present ALL retrieved jobs with all 35 available details (Title, Company, Industry, Size, Location, Remote/Hybrid/Onsite, Experience, Skills, Salary, Dates, Contacts, URL, Description).
   - Perform a thorough JD vs Resume comparative analysis for each job.
   - Provide a definitive RANKING (Rank #1 to #N) with concrete reasons.
   - Provide Skill Gap Analysis and Recruiter Outreach Tips.
   - Conclude with proactive application offer:
     "💡 **Next Step**: If you wish, I can draft a personalized application email and tailored cover letter for this top role (or any of the above jobs) and prepare it for immediate submission. Would you like to proceed with an application? (Simply say: 'Apply for Job 1' or 'Apply to this role')"
2. If 0 jobs were found:
   - Clearly inform the user that 0 jobs in the database match their query.
   - Do NOT fabricate imaginary jobs.
   - Advise them to click '🔄 Collect New Jobs (Exa Agent)' near the input bar to fetch fresh jobs from Exa.ai, or broaden their search criteria.
""",
            expected_output="A complete, professional markdown report presenting all jobs, comparative ranking, and next steps.",
            agent=advisor_agent,
            context=[retrieval_task],
        )

        crew = Crew(
            agents=[job_agent, advisor_agent],
            tasks=[retrieval_task, matching_task],
            process=Process.sequential,
            verbose=False,
        )

        self.notify("💼 Talent Strategist & Career Advisor: Collaborating on resume matching & ranking...")
        result = crew.kickoff()
        response_text = str(result.raw if hasattr(result, "raw") else result)

        tool_ctx = get_tool_context()
        retrieved = tool_ctx.get("last_retrieved_jobs", [])
        self.state.retrieved_jobs = retrieved
        if retrieved:
            self.state.cached_jobs = retrieved

        self.state.meta_filters = {"search_query": rewritten_query}
        self.state.final_response = response_text
        return response_text

    # ------------------------------------------------------------
    # Branch B: Web Intelligence Crew
    # ------------------------------------------------------------
    @listen("execute_web_search")
    def handle_web_search_crew(self):
        """
        Company Intelligence Researcher uses Exa web search to gather live market data.
        """
        self.notify("🌐 Company Intelligence Agent: Gathering live online intelligence via Exa...")
        response_text = run_company_intelligence_crew(
            user_query=self.state.user_query,
            history_summary=self.state.history_summary,
            model_name=self.state.model_name,
        )
        self.state.meta_filters = {"web_search_query": self.state.user_query}
        self.state.final_response = response_text
        return response_text

    # ------------------------------------------------------------
    # Branch C: Job Application & Recruiter Outreach Crew
    # ------------------------------------------------------------
    @listen("execute_job_application")
    def handle_job_application_crew(self):
        """
        Application & Recruiter Outreach Agent resolves target contacts and drafts tailored cover letters.
        """
        self.notify("📝 Application Specialist Agent: Resolving target contacts & drafting cover letter...")
        app_res = run_job_application_crew(
            user_query=self.state.user_query,
            resume_text=self.state.resume_text,
            cached_jobs=self.state.cached_jobs,
            collection_name=self.state.collection_name,
            model_name=self.state.model_name,
        )
        self.state.email_draft = app_res.get("email_draft")
        if app_res.get("target_job"):
            self.state.retrieved_jobs = [app_res["target_job"]]
        self.state.final_response = app_res["final_text"]
        return app_res["final_text"]

    # ------------------------------------------------------------
    # Branch D: General Career Advisory Crew
    # ------------------------------------------------------------
    @listen("execute_general_advisory")
    def handle_general_advisory_crew(self):
        """
        Senior LinkedIn Career Advisor provides mentoring, resume tips, or compares active jobs.
        """
        self.notify("🤝 Career Advisor Agent: Preparing career guidance & comparative evaluation...")
        response_text = run_general_career_crew(
            user_query=self.state.user_query,
            resume_text=self.state.resume_text,
            cached_jobs=self.state.cached_jobs,
            history_context=self.state.history_context,
            model_name=self.state.model_name,
        )
        self.state.final_response = response_text
        return response_text


# ============================================================
# Main Turn Execution Entrypoint
# ============================================================

def execute_crewai_turn(
    user_query: str,
    resume_text: Optional[str] = None,
    thread_id: str = "default_session",
    project_id: Optional[str] = None,
    project_name: Optional[str] = None,
    collection_name: str = "linkedin_jobs",
    model_name: str = DEFAULT_MODEL,
    status_callback: Optional[Callable[[str], None]] = None,
    chat_history: Optional[List[Dict[str, Any]]] = None,
    current_jobs: Optional[List[Dict[str, Any]]] = None,
    history_summary: Optional[str] = None,
    conversation_title: Optional[str] = "New Chat",
) -> Dict[str, Any]:
    """
    Executes a complete conversational turn using the CrewAI CareerAssistantFlow:
    1. Sets up tool context for the active project and session.
    2. Initializes CareerAssistantFlow with session state.
    3. Runs CrewAI flow kickoff (triggering agents, tasks, and crews).
    4. Yields streamed output chunks back to Streamlit and Telegram.
    """
    cached_jobs = current_jobs or []
    messages = chat_history or []

    # Configure tool runtime context
    set_tool_context(
        collection_name=collection_name,
        cached_jobs=cached_jobs,
        resume_text=resume_text,
    )

    # Initialize Flow
    flow = CareerAssistantFlow(status_callback=status_callback)
    flow.state.user_query = user_query
    flow.state.thread_id = thread_id
    flow.state.resume_text = resume_text
    flow.state.messages = messages
    flow.state.history_summary = history_summary
    flow.state.cached_jobs = cached_jobs
    flow.state.project_id = project_id
    flow.state.project_name = project_name
    flow.state.collection_name = collection_name
    flow.state.conversation_title = conversation_title or "New Chat"
    flow.state.model_name = model_name

    # Kickoff CrewAI Multi-Agent Flow
    flow.kickoff()

    # Safety guarantee: If flow listener did not populate final_response, dispatch directly
    if not flow.state.final_response:
        intent = flow.state.intent or "general"
        if intent == "job_search":
            flow.state.final_response = flow.handle_job_search_crew()
        elif intent == "web_search":
            flow.state.final_response = flow.handle_web_search_crew()
        elif intent == "apply_job":
            flow.state.final_response = flow.handle_job_application_crew()
        else:
            flow.state.final_response = flow.handle_general_advisory_crew()

    final_text = flow.state.final_response
    # Generator streaming words in chunks for responsive UI
    def _response_stream():
        words = final_text.split(" ")
        for i in range(0, len(words), 3):
            yield " ".join(words[i:i+3]) + " "

    return {
        "intent": flow.state.intent,
        "history_summary": flow.state.history_summary,
        "current_jobs": flow.state.cached_jobs,
        "retrieved_jobs": flow.state.retrieved_jobs,
        "search_query": flow.state.search_query,
        "meta_filters": flow.state.meta_filters,
        "email_draft": flow.state.email_draft,
        "response_stream": _response_stream(),
        "final_text": final_text,
    }
