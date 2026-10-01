"""
Node: Token-Based History Check & Summarization.
Enforces a 150k token pool for active conversation history.
If dialogue exceeds 150k tokens, progressively summarizes older turns
(first query + response, then second, etc.) into a running summary.
"""
from __future__ import annotations

from typing import Dict, Any, List, Optional
import ollama

DEFAULT_MODEL = "bjoernb/gemma4-31b-fast:latest"
WELCOME_BANNER = "👋 Hello! I am your **LinkedIn AI Job & Career Advisor**"
MAX_TOKEN_POOL = 150000  # 150k tokens budget for active conversation history

_tokenizer = None


def get_tokenizer():
    """Lazy initialization of tiktoken cl100k_base tokenizer."""
    global _tokenizer
    if _tokenizer is None:
        try:
            import tiktoken
            _tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception:
            _tokenizer = None
    return _tokenizer


def count_tokens(text: str) -> int:
    """Calculate exact or estimated token count for text."""
    if not text:
        return 0
    enc = get_tokenizer()
    if enc:
        try:
            return len(enc.encode(str(text)))
        except Exception:
            pass
    # Fallback heuristic: 1 token ~= 4 chars
    return max(1, len(str(text)) // 4)


def count_message_tokens(messages: List[Dict[str, Any]]) -> int:
    """Sum total tokens across a list of message dicts."""
    total = 0
    for m in messages:
        c = m.get("content", "")
        total += count_tokens(str(c)) + 4
    return total


def summarize_older_history(
    older_messages: List[Dict[str, str]],
    existing_summary: Optional[str] = None,
    model_name: str = DEFAULT_MODEL,
) -> str:
    """
    Call Ollama LLM to summarize conversation history older than the token pool.
    Preserves user profile, preferences, discussed jobs, and key context.
    """
    if not older_messages:
        return existing_summary or ""

    formatted = []
    for m in older_messages:
        role = m.get("role", "user").capitalize()
        content = m.get("content", "")
        if content:
            formatted.append(f"{role}: {content}")

    dialogue_text = "\n".join(formatted)

    prompt = f"""You are an expert conversation context summarizer.
Summarize the key facts, user requests, career goals, mentioned job titles/companies, and decisions from this earlier chat history.
Keep the summary informative and accurate (under 10000 words).

Earlier Conversation:
{dialogue_text}
"""
    if existing_summary:
        prompt += f"\nPrevious Running Summary to incorporate:\n{existing_summary}\n"

    prompt += "\nOutput ONLY the summary text, without any conversational filler or introductory phrases:"

    try:
        response = ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.2, "num_predict": 11000},
        )
        summary = response.get("message", {}).get("content", "").strip()
        return summary if summary else (existing_summary or "")
    except Exception as e:
        print(f"[WARN] Dialogue history summarization failed: {e}")
        return existing_summary or ""


# ------------------------------------------------------------------
# Conditional Edge Function
# ------------------------------------------------------------------

def check_history_condition(state: AgentState) -> str:
    """
    Conditional edge: Routes to 'summarize_history' if dialogue exceeds 150k tokens,
    else routes to 'skip_summary'.
    """
    messages = state.get("messages", [])
    filtered = [m for m in messages if WELCOME_BANNER not in str(m.get("content", ""))]
    total_tokens = count_message_tokens(filtered)

    if total_tokens > MAX_TOKEN_POOL:
        return "summarize_history"
    return "skip_summary"


# ------------------------------------------------------------------
# Graph Nodes
# ------------------------------------------------------------------

def summarize_history_node(state: AgentState) -> Dict[str, Any]:
    """
    Node: Progressively summarizes older turns (1st query+response, then 2nd, etc.)
    from the beginning of chat history until the remaining active dialogue
    fits inside the 150k token pool.
    """
    messages = state.get("messages", [])
    filtered = [m for m in messages if WELCOME_BANNER not in str(m.get("content", ""))]
    cb = state.get("status_callback")
    if cb:
        cb("Summarizing older conversation turns (150k token pool limit)...")

    model_name = state.get("model_name", DEFAULT_MODEL)
    current_summary = state.get("history_summary") or ""

    remaining = list(filtered)
    # Target headroom: bring active tokens down to ~120k to avoid summarizing on every single turn
    target_pool = int(MAX_TOKEN_POOL * 0.8)

    # Progressively summarize oldest turns (query + response pair)
    while count_message_tokens(remaining) > target_pool and len(remaining) > 2:
        # Take the oldest turn: usually 1 user query and 1 assistant response
        turn_to_summarize = []
        if remaining and remaining[0].get("role") == "user":
            turn_to_summarize.append(remaining.pop(0))
            if remaining and remaining[0].get("role") == "assistant":
                turn_to_summarize.append(remaining.pop(0))
        else:
            # If unexpected ordering, pop at least one message
            turn_to_summarize.append(remaining.pop(0))

        current_summary = summarize_older_history(
            older_messages=turn_to_summarize,
            existing_summary=current_summary,
            model_name=model_name,
        )

    return {
        "history_summary": current_summary,
        "recent_messages": remaining,
    }


def passthrough_history_node(state: AgentState) -> Dict[str, Any]:
    """Node: Passes through dialogue history without summarization when <= 150k tokens."""
    messages = state.get("messages", [])
    filtered = [m for m in messages if WELCOME_BANNER not in str(m.get("content", ""))]
    return {
        "recent_messages": filtered,
        "history_summary": state.get("history_summary"),
    }
