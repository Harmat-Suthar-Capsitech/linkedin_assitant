import json
import os
import re
from datetime import datetime
from typing import Optional, Dict, Any, List

LOGS_DIR = "logs"


def safe_parse_json(text: str) -> Dict[str, Any]:
    """
    Safely extract and parse JSON even if wrapped in markdown code fences
    (e.g., ```json ... ```) or preceded/followed by explanatory text.
    """
    if not text:
        return {}
    cleaned = text.strip()
    if "```" in cleaned:
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned, re.IGNORECASE)
        if match:
            cleaned = match.group(1).strip()
    if not (cleaned.startswith("{") and cleaned.endswith("}")):
        match = re.search(r"\{[\s\S]*\}", cleaned)
        if match:
            cleaned = match.group(0).strip()
    try:
        return json.loads(cleaned)
    except Exception:
        return {}


def clean_number(val: Any) -> Optional[float]:
    """Helper to sanitize numeric values from LLM output (e.g. '50k', '$120,000')."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        val_clean = val.lower().replace(",", "").replace("$", "").replace("₹", "").strip()
        if "k" in val_clean:
            try:
                num = float(val_clean.replace("k", "").strip())
                return num * 1000
            except ValueError:
                pass
        try:
            match = re.search(r"[-+]?\d*\.?\d+", val_clean)
            if match:
                return float(match.group())
        except Exception:
            pass
    return None


def sanitize_filename(name: str, fallback: str = "conversation") -> str:
    """Sanitize arbitrary string into safe filesystem name."""
    if not name:
        return fallback
    clean = re.sub(r'[^a-zA-Z0-9_\- ]', '', str(name)).strip()
    clean = clean.replace(" ", "_")[:60]
    return clean or fallback


def save_pipeline_log(
    thread_id: str,
    user_query: str,
    intent: str,
    meta_filters: Optional[Dict[str, Any]] = None,
    qdrant_results: Optional[List[Dict[str, Any]]] = None,
    resume_text: Optional[str] = None,
    history_summary: Optional[str] = None,
    project_name: Optional[str] = None,
    conversation_title: Optional[str] = None,
    exa_metrics: Optional[Dict[str, Any]] = None,
    search_inputs: Optional[Dict[str, Any]] = None,
    assistant_response: Optional[str] = None,
):
    """
    Save structured pipeline logs for each conversation turn.
    Stored inside: logs/<project_name>/<conversation_title>.txt
    Includes:
    - Project Name
    - Job Find Inputs & Criteria
    - Exa output metrics (time, cost in dollars, job count)
    - Query turn details (user query, intent, filters, results)
    - Full extracted resume text
    - Full chatbot response
    """
    try:
        # Determine project folder
        clean_proj = sanitize_filename(project_name, fallback="default_project")
        project_log_dir = os.path.join(LOGS_DIR, clean_proj)
        os.makedirs(project_log_dir, exist_ok=True)

        # Determine conversation filename from title
        clean_title = sanitize_filename(conversation_title, fallback=f"chat_{thread_id[:8]}")
        log_file_path = os.path.join(project_log_dir, f"{clean_title}.txt")

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_lines = []
        log_lines.append("=" * 80)
        log_lines.append(f"PIPELINE LOG — {timestamp}")
        log_lines.append(f"Project: {project_name or clean_proj} | Thread ID: {thread_id}")
        log_lines.append("=" * 80)

        # Exa / Job search input & metrics (if recorded for this project/turn)
        if search_inputs or exa_metrics:
            log_lines.append("\n" + "-" * 40)
            log_lines.append("🤖 EXA.AI AGENT RESEARCH & METRICS")
            log_lines.append("-" * 40)
            if search_inputs:
                log_lines.append(f"Search Inputs / Criteria: {json.dumps(search_inputs, indent=2)}")
            if exa_metrics:
                log_lines.append(f"Run ID        : {exa_metrics.get('run_id')}")
                log_lines.append(f"Execution Time: {exa_metrics.get('duration_seconds')}s")
                log_lines.append(f"Cost (Dollars): ${exa_metrics.get('cost_dollars', 0.0):.4f}")
                log_lines.append(f"Jobs Collected: {exa_metrics.get('job_count', 0)}")
                log_lines.append(f"Status        : {exa_metrics.get('status', 'completed')}")

        log_lines.append(f"\n📝 User Query     : {user_query}")
        log_lines.append(f"🎯 Detected Intent: {intent}")

        # Meta filters
        log_lines.append("\n" + "-" * 40)
        log_lines.append("⚙️ EXTRACTED META FILTERS")
        log_lines.append("-" * 40)
        if meta_filters:
            log_lines.append(json.dumps(meta_filters, indent=2))
        else:
            log_lines.append("(No filters applicable for this turn)")

        # Chat history summary
        log_lines.append("\n" + "-" * 40)
        log_lines.append("🧠 CHAT HISTORY SUMMARY")
        log_lines.append("-" * 40)
        if history_summary:
            log_lines.append(history_summary)
        else:
            log_lines.append("(No older history summary — dialogue within active sliding window)")

        # Resume text (Full extracted text for debugging)
        if resume_text:
            log_lines.append("\n" + "-" * 40)
            log_lines.append("📄 ATTACHED RESUME FULL EXTRACTED TEXT")
            log_lines.append("-" * 40)
            log_lines.append(str(resume_text).strip())

        # Qdrant top results
        log_lines.append("\n" + "-" * 40)
        if qdrant_results is not None:
            if len(qdrant_results) > 0:
                log_lines.append(f"🔍 QDRANT RETRIEVED JOBS ({len(qdrant_results)} Candidate Jobs)")
                log_lines.append("-" * 40)
                for job in qdrant_results:
                    rank = job.get("rank", "?")
                    score = job.get("score", "N/A")
                    title = job.get("jobTitle") or job.get("Job Title")
                    company = job.get("company") or job.get("Company")
                    city = job.get("city") or job.get("Location")
                    email = job.get("companyEmail") or job.get("Company Email")
                    url = job.get("sourceUrl") or job.get("Source URL")
                    log_lines.append(f"\n  #{rank} [{score}] {title} at {company} ({city})")
                    if email:
                        log_lines.append(f"    Email     : {email}")
                    if url:
                        log_lines.append(f"    Source URL: {url}")
            else:
                log_lines.append("🔍 QDRANT RETRIEVAL RESULTS (0 Candidate Jobs Found)")
                log_lines.append("-" * 40)
                log_lines.append("(Zero jobs matched the query/filters in Qdrant database)")
        else:
            log_lines.append("🔍 QDRANT RETRIEVAL RESULTS")
            log_lines.append("-" * 40)
            log_lines.append(f"(No Qdrant retrieval performed for this turn — intent: {intent})")

        # Chatbot full response
        if assistant_response:
            log_lines.append("\n" + "-" * 40)
            log_lines.append("🤖 CHATBOT FULL RESPONSE")
            log_lines.append("-" * 40)
            log_lines.append(str(assistant_response).strip())

        log_lines.append("\n" + "=" * 80 + "\n\n")
        log_content = "\n".join(log_lines)

        # Write to project-specific conversation log file
        with open(log_file_path, "a", encoding="utf-8") as f:
            f.write(log_content)

    except Exception as e:
        print(f"[WARN] Failed to write pipeline log: {e}")
