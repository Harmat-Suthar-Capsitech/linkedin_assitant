"""
LinkedIn Career Assistant — CrewAI Multi-Agent System.
Replaces LangGraph with CrewAI multi-agent orchestration across specialized agents:
- History & Memory Agent
- Intake & Routing Coordinator Agent
- Job Retrieval & Matching Strategist Agent (35 schema fields)
- Company Intelligence Researcher Agent
- Job Application & Outreach Specialist Agent
- Senior Career Advisor Agent

Provides the unified stream_assistant_turn() entry point used by Streamlit and Telegram.
"""
from __future__ import annotations

from typing import List, Dict, Any, Optional, Generator, Callable
from dotenv import load_dotenv

# CrewAI Multi-Agent Workflow Engine
from crew_workflow import (
    execute_crewai_turn,
    classify_intent_with_router_agent,
    run_job_application_crew,
    run_company_intelligence_crew,
    run_general_career_crew,
)
from crew_agents import (
    DEFAULT_MODEL,
    # create_history_agent,
    # create_router_agent,
    # create_query_rewriter_agent,
    # create_job_retrieval_agent,
    # create_company_intelligence_agent,
    # create_job_application_agent,
    # create_general_career_agent,
)

# Session persistence
from redis_store import get_session_state, save_session_state, get_project
from logger_util import save_pipeline_log

# Re-exports used by app.py
from retrieve_jobs import ALL_SCHEMA_COLUMNS

load_dotenv()


# ============================================================
# Main Entrypoint: Unified Streaming Assistant Runner via CrewAI
# ============================================================

def stream_assistant_turn(
    user_query: str,
    resume_text: Optional[str] = None,
    thread_id: str = "default_session",
    collection_id: Optional[str] = None,
    project_id: Optional[str] = None,
    model_name: str = DEFAULT_MODEL,
    status_callback: Optional[Callable[[str], None]] = None,
    chat_history: Optional[List[Dict[str, str]]] = None,
) -> Generator[str, None, None]:
    """
    Unified streaming runner executing CrewAI Multi-Agent Orchestration:
    1. Loads persistent session state from Redis.
    2. Identifies collection and dedicated Qdrant vector store.
    3. Invokes CrewAI workflow (History Agent -> Routing Agent -> Specialist Agent Crew).
    4. Streams response tokens back in real-time.
    5. Saves updated persistent session memory to Redis.
    6. Logs turn details to conversation log file.
    """
    session = get_session_state(thread_id)
    current_jobs = session.get("current_jobs", [])
    history_summary = session.get("history_summary", None)

    # Resolve collection metadata (backward compat with project_id)
    active_cid = collection_id or project_id or session.get("collection_id") or session.get("project_id")
    collection_title = None
    qdrant_collection_name = "linkedin_jobs"
    if active_cid:
        coll = get_project(active_cid)
        if coll:
            collection_title = coll.get("name")
            qdrant_collection_name = coll.get("qdrant_collection") or coll.get("collection_name", "linkedin_jobs")

    # Maintain resume in session
    if resume_text:
        session["resume_text"] = resume_text
    else:
        resume_text = session.get("resume_text")

    # Message history from caller or session
    if chat_history is not None:
        messages = [m for m in chat_history if isinstance(m, dict)]
    else:
        messages = session.get("messages", [])

    # Execute multi-agent orchestration turn
    crew_result = execute_crewai_turn(
        user_query=user_query,
        resume_text=resume_text,
        thread_id=thread_id,
        project_id=active_cid,
        project_name=collection_title,
        collection_name=qdrant_collection_name,
        model_name=model_name,
        status_callback=status_callback,
        chat_history=messages,
        current_jobs=current_jobs,
        history_summary=history_summary,
        conversation_title=session.get("title", "New Chat"),
    )

    # Stream the tokens in real time
    stream = crew_result.get("response_stream")
    full_response_text = ""
    if stream:
        try:
            for chunk in stream:
                delta = ""
                # 1. Ollama ChatResponse object
                msg = getattr(chunk, "message", None)
                if msg is not None:
                    delta = getattr(msg, "content", "") if not isinstance(msg, dict) else msg.get("content", "")
                # 2. Dictionary format
                elif isinstance(chunk, dict):
                    if "message" in chunk and isinstance(chunk["message"], dict):
                        delta = chunk["message"].get("content", "")
                    else:
                        delta = chunk.get("content", "")
                # 3. CrewStreamingOutput or object with content attribute
                elif hasattr(chunk, "content"):
                    delta = chunk.content
                # 4. String or fallback
                elif isinstance(chunk, str):
                    delta = chunk
                
                if delta:
                    full_response_text += delta
                    yield delta
        except Exception as e:
            err = f"\n\n[ERROR] Streaming error: {e}"
            full_response_text += err
            yield err

    if not full_response_text.strip():
        fallback = "Hello! I am your LinkedIn AI Career Assistant. How can I assist you with your career or job search today?"
        full_response_text += fallback
        yield fallback

    # Update session state and persist to Redis
    updated_messages = list(messages) + [
        {"role": "user", "content": user_query},
        {"role": "assistant", "content": full_response_text},
    ]
    session["messages"] = updated_messages
    session["user_query"] = user_query
    session["current_jobs"] = crew_result.get("current_jobs", current_jobs)
    session["intent"] = crew_result.get("intent", "general")
    session["history_summary"] = crew_result.get("history_summary")
    if crew_result.get("email_draft") is not None:
        session["email_draft"] = crew_result["email_draft"]

    save_session_state(thread_id, session, collection_id=active_cid)

    # Save comprehensive pipeline log
    try:
        save_pipeline_log(
            thread_id=thread_id,
            user_query=user_query,
            intent=crew_result.get("intent", "general"),
            meta_filters=crew_result.get("meta_filters"),
            qdrant_results=crew_result.get("retrieved_jobs"),
            resume_text=resume_text,
            history_summary=crew_result.get("history_summary"),
            project_name=collection_title,
            conversation_title=session.get("title", "New Chat"),
            assistant_response=full_response_text,
        )
    except Exception as e:
        print(f"[WARN] Failed to write pipeline log at turn end: {e}")


# ============================================================
# Standalone Intent Detection Node (used by app.py / collection creation)
# ============================================================

def intent_detection_node(state: Dict[str, Any]) -> Dict[str, str]:
    """
    Intake intent classification node delegating directly to the CrewAI
    Intake & Routing Coordinator Agent (create_router_agent) in crew_workflow.py.
    Returns: {"intent": "job_search" | "web_search" | "apply_job" | "general"}
    """
    user_query = str(state.get("user_query") or "").strip()
    model_name = state.get("model_name", DEFAULT_MODEL)
    history_summary = state.get("history_summary")
    recent_messages = state.get("recent_messages", [])
    has_cached_jobs = bool(state.get("has_cached_jobs", False))

    history_parts = []
    if history_summary:
        history_parts.append(f"Summary: {history_summary}")
    for m in recent_messages[-4:]:
        role = m.get("role", "user").capitalize()
        c = str(m.get("content", ""))[:200]
        if c:
            history_parts.append(f"{role}: {c}")
    history_context = "\n".join(history_parts)

    intent = classify_intent_with_router_agent(
        user_query=user_query,
        history_context=history_context,
        has_cached_jobs=has_cached_jobs,
        model_name=model_name,
    )
    return {"intent": intent}
