"""
Redis-backed persistent session store for LinkedIn Assistant.
Supports:
1. Collections (name, search criteria, Exa metrics, Qdrant vector store per collection)
2. Collection-scoped conversations & persistent chat history
3. Active jobs, resume text, email drafts, and history summaries per thread
Data survives system restarts via Redis AOF persistence.
"""
from __future__ import annotations

import json
import os
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional

import redis
from dotenv import load_dotenv

load_dotenv()

REDIS_URL = os.getenv("REDIS_URL")
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD") or os.getenv("REDIS_API_KEY") or None
REDIS_SSL = os.getenv("REDIS_SSL", "").strip().lower() in ("true", "1", "yes")
REDIS_PREFIX = "linkedin_assistant"

# Key TTL (30 days) — conversations expire after 30 days of inactivity
KEY_TTL_SECONDS = 30 * 24 * 60 * 60


def _get_redis() -> redis.Redis:
    """Return a Redis client instance supporting both Cloud (URL/SSL/Auth) and Localhost."""
    redis_url = os.getenv("REDIS_URL")
    if redis_url:
        return redis.from_url(
            redis_url,
            decode_responses=True,
            socket_connect_timeout=10,
            socket_timeout=10,
            retry_on_timeout=True,
        )

    host = os.getenv("REDIS_HOST", "localhost")
    port = int(os.getenv("REDIS_PORT", 6379))
    db = int(os.getenv("REDIS_DB", 0))
    password = os.getenv("REDIS_PASSWORD") or os.getenv("REDIS_API_KEY") or None
    ssl_env = os.getenv("REDIS_SSL", "").strip().lower()
    ssl = (ssl_env in ("true", "1", "yes")) or (port in (6380, 25061)) or ("upstash" in host.lower())

    return redis.Redis(
        host=host,
        port=port,
        db=db,
        password=password,
        ssl=ssl,
        decode_responses=True,
        socket_connect_timeout=10,
        socket_timeout=10,
        retry_on_timeout=True,
    )


# ------------------------------------------------------------------
# Key Builders
# ------------------------------------------------------------------

def _thread_key(thread_id: str) -> str:
    """Redis key for a specific conversation thread."""
    return f"{REDIS_PREFIX}:thread:{thread_id}"


def _conversations_set_key() -> str:
    """Redis key for the set that tracks all conversation thread IDs."""
    return f"{REDIS_PREFIX}:conversations"


def _collections_set_key() -> str:
    """Redis key for tracking all collection IDs."""
    return f"{REDIS_PREFIX}:projects"


def _collection_key(collection_id: str) -> str:
    """Redis key for a specific collection metadata."""
    return f"{REDIS_PREFIX}:project:{collection_id}"


def _collection_conversations_key(collection_id: str) -> str:
    """Redis key for the set of conversation thread IDs belonging to a collection."""
    return f"{REDIS_PREFIX}:project:{collection_id}:conversations"


# ------------------------------------------------------------------
# Collection CRUD
# ------------------------------------------------------------------

def create_collection(
    name: str,
    search_query: str,
    search_metadata: Optional[Dict[str, Any]] = None,
    exa_metrics: Optional[Dict[str, Any]] = None,
    qdrant_collection: Optional[str] = None,
) -> str:
    """
    Create a new collection with its job search criteria and initial Exa metrics.
    Returns collection_id.
    """
    collection_id = str(uuid.uuid4())
    clean_id = collection_id.replace("-", "_")[:12]
    assigned_qdrant = qdrant_collection or f"jobs_proj_{clean_id}"

    collection_data = {
        "collection_id": collection_id,
        "name": name.strip() or f"Collection {clean_id}",
        "search_query": search_query,
        "search_metadata": search_metadata or {},
        "qdrant_collection": assigned_qdrant,
        # Backward compat: keep old key too so existing Redis data doesn't break
        "collection_name": assigned_qdrant,
        "created_at": datetime.now().isoformat(),
        "exa_metrics": exa_metrics or {},
        "total_jobs": (exa_metrics or {}).get("job_count", 0),
    }

    try:
        r = _get_redis()
        r.set(_collection_key(collection_id), json.dumps(collection_data, default=str))
        r.sadd(_collections_set_key(), collection_id)
    except Exception as e:
        print(f"[WARN] Redis create_collection failed: {e}")

    return collection_id


def get_collection(collection_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve metadata for a specific collection."""
    try:
        r = _get_redis()
        raw = r.get(_collection_key(collection_id))
        if raw:
            data = json.loads(raw)
            # Normalize: ensure both collection_id and qdrant_collection are present
            if "collection_id" not in data and "project_id" in data:
                data["collection_id"] = data["project_id"]
            if "qdrant_collection" not in data and "collection_name" in data:
                data["qdrant_collection"] = data["collection_name"]
            return data
    except Exception as e:
        print(f"[WARN] Redis get_collection failed: {e}")
    return None


def list_collections() -> List[Dict[str, Any]]:
    """Return all collections sorted newest first."""
    try:
        r = _get_redis()
        collection_ids = r.smembers(_collections_set_key())
        collections = []
        for cid in collection_ids:
            raw = r.get(_collection_key(cid))
            if raw:
                coll = json.loads(raw)
                if "collection_id" not in coll and "project_id" in coll:
                    coll["collection_id"] = coll["project_id"]
                if "qdrant_collection" not in coll and "collection_name" in coll:
                    coll["qdrant_collection"] = coll["collection_name"]
                # Count conversations for this collection
                conv_count = r.scard(_collection_conversations_key(cid))
                coll["conversation_count"] = conv_count
                collections.append(coll)
            else:
                r.srem(_collections_set_key(), cid)

        collections.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return collections
    except Exception as e:
        print(f"[WARN] Redis list_collections failed: {e}")
        return []


def update_collection(collection_id: str, updates: Dict[str, Any]):
    """Update fields on an existing collection (e.g. exa_metrics, total_jobs)."""
    try:
        r = _get_redis()
        raw = r.get(_collection_key(collection_id))
        if raw:
            coll = json.loads(raw)
            coll.update(updates)
            r.set(_collection_key(collection_id), json.dumps(coll, default=str))
    except Exception as e:
        print(f"[WARN] Redis update_collection failed: {e}")


def delete_collection(collection_id: str):
    """Delete a collection, all its conversations, and remove its Qdrant vector store."""
    try:
        r = _get_redis()
        # 1. Delete all associated conversations
        conv_ids = r.smembers(_collection_conversations_key(collection_id))
        for cid in conv_ids:
            delete_session(cid)
        r.delete(_collection_conversations_key(collection_id))

        # 2. Get Qdrant collection to delete from Qdrant if needed
        coll = get_collection(collection_id)
        qdrant_name = None
        if coll:
            qdrant_name = coll.get("qdrant_collection") or coll.get("collection_name")
        if qdrant_name:
            try:
                from qdrant_client import QdrantClient
                qc = QdrantClient(
                    url=os.getenv("QDRANT_URL", "http://localhost:6333"),
                    api_key=os.getenv("QDRANT_API_KEY") or None,
                    check_compatibility=False,
                )
                qc.delete_collection(qdrant_name)
            except Exception as qe:
                print(f"[INFO] Qdrant delete collection cleanup: {qe}")

        # 3. Delete collection key and unregister from set
        r.delete(_collection_key(collection_id))
        r.srem(_collections_set_key(), collection_id)
    except Exception as e:
        print(f"[WARN] Redis delete_collection failed: {e}")


# Backward-compat aliases so any old code still calling these works
create_project = create_collection
get_project = get_collection
list_projects = list_collections
update_project = update_collection
delete_project = delete_collection


# ------------------------------------------------------------------
# Session State CRUD (Conversations)
# ------------------------------------------------------------------

def get_session_state(thread_id: str) -> Dict[str, Any]:
    """
    Load session state from Redis for the given thread_id.
    Returns state dict. If thread_id doesn't exist, returns fresh state.
    """
    try:
        r = _get_redis()
        key = _thread_key(thread_id)
        raw = r.get(key)
        if raw:
            state = json.loads(raw)
            state.setdefault("messages", [])
            state.setdefault("current_jobs", [])
            state.setdefault("resume_text", None)
            state.setdefault("resume_filename", None)
            state.setdefault("resume_file_path", None)
            state.setdefault("email_draft", None)
            state.setdefault("history_summary", None)
            state.setdefault("title", "New Chat")
            state.setdefault("collection_id", state.get("project_id"))
            state.setdefault("created_at", datetime.now().isoformat())
            return state
    except Exception as e:
        print(f"[WARN] Redis read failed for thread {thread_id}: {e}")

    return {
        "messages": [],
        "current_jobs": [],
        "resume_text": None,
        "resume_filename": None,
        "resume_file_path": None,
        "email_draft": None,
        "history_summary": None,
        "title": "New Chat",
        "collection_id": None,
        "created_at": datetime.now().isoformat(),
    }


def save_session_state(thread_id: str, state: Dict[str, Any], collection_id: Optional[str] = None):
    """
    Save session state to Redis for the given thread_id.
    Also associates with collection_id and adds to conversation sets.
    """
    try:
        r = _get_redis()
        key = _thread_key(thread_id)

        # Attach collection_id if specified
        if collection_id:
            state["collection_id"] = collection_id
            state["project_id"] = collection_id  # backward compat
        current_cid = state.get("collection_id") or state.get("project_id")

        # Auto-generate title from first user message if still "New Chat"
        if state.get("title", "New Chat") == "New Chat":
            for m in state.get("messages", []):
                if m.get("role") == "user" and m.get("content"):
                    title = str(m["content"]).strip()[:60]
                    if len(str(m["content"]).strip()) > 60:
                        title += "..."
                    state["title"] = title
                    break

        r.set(key, json.dumps(state, default=str), ex=KEY_TTL_SECONDS)
        r.sadd(_conversations_set_key(), thread_id)

        # Track conversation under collection if associated
        if current_cid:
            r.sadd(_collection_conversations_key(current_cid), thread_id)
    except Exception as e:
        print(f"[WARN] Redis write failed for thread {thread_id}: {e}")


def delete_session(thread_id: str):
    """Delete a conversation from Redis and remove from collection sets."""
    try:
        r = _get_redis()
        state = get_session_state(thread_id)
        cid = state.get("collection_id") or state.get("project_id")
        if cid:
            r.srem(_collection_conversations_key(cid), thread_id)

        r.delete(_thread_key(thread_id))
        r.srem(_conversations_set_key(), thread_id)
    except Exception as e:
        print(f"[WARN] Redis delete failed for thread {thread_id}: {e}")


def list_conversations(collection_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Return a list of conversation threads with metadata.
    If collection_id is provided, returns ONLY conversations for that collection.
    Sorted by created_at descending.
    """
    try:
        r = _get_redis()
        if collection_id:
            thread_ids = r.smembers(_collection_conversations_key(collection_id))
        else:
            thread_ids = r.smembers(_conversations_set_key())

        conversations = []
        for tid in thread_ids:
            raw = r.get(_thread_key(tid))
            if raw:
                state = json.loads(raw)
                raw_title = str(state.get("title") or "New Chat").strip()
                # Truncate title so it fits cleanly inside the sidebar button box (~20-22 chars)
                display_title = (raw_title[:20] + "...") if len(raw_title) > 23 else raw_title

                conversations.append({
                    "thread_id": tid,
                    "title": display_title,
                    "full_title": raw_title,
                    "collection_id": state.get("collection_id") or state.get("project_id"),
                    "created_at": state.get("created_at", ""),
                    "message_count": len(state.get("messages", [])),
                })
            else:
                if collection_id:
                    r.srem(_collection_conversations_key(collection_id), tid)
                r.srem(_conversations_set_key(), tid)

        conversations.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return conversations
    except Exception as e:
        print(f"[WARN] Redis list conversations failed: {e}")
        return []


def clear_all_conversations(collection_id: Optional[str] = None):
    """Delete all conversations (or all conversations in a specific collection)."""
    try:
        r = _get_redis()
        if collection_id:
            thread_ids = r.smembers(_collection_conversations_key(collection_id))
        else:
            thread_ids = r.smembers(_conversations_set_key())

        for tid in thread_ids:
            delete_session(tid)

        if collection_id:
            r.delete(_collection_conversations_key(collection_id))
    except Exception as e:
        print(f"[WARN] Redis clear all conversations failed: {e}")


# ------------------------------------------------------------------
# Multi-User Telegram Scoped Storage Helpers
# ------------------------------------------------------------------

def get_user_active_collection(user_id: str | int) -> Optional[str]:
    """Retrieve active collection_id for a specific Telegram user."""
    uid = str(user_id)
    try:
        r = _get_redis()
        cid = r.get(f"{REDIS_PREFIX}:user:{uid}:active_project")
        if cid and get_collection(cid):
            return cid
        # Fallback to the latest available collection if user has any
        user_colls = list_user_collections(uid)
        if user_colls:
            latest_cid = user_colls[0]["collection_id"]
            set_user_active_collection(uid, latest_cid)
            return latest_cid
    except Exception as e:
        print(f"[WARN] Redis get_user_active_collection failed: {e}")
        return None
    return None


def set_user_active_collection(user_id: str | int, collection_id: str):
    """Set active collection_id for a specific Telegram user."""
    uid = str(user_id)
    try:
        r = _get_redis()
        r.set(f"{REDIS_PREFIX}:user:{uid}:active_project", collection_id)
    except Exception as e:
        print(f"[WARN] Redis set_user_active_collection failed: {e}")


def get_user_active_thread(user_id: str | int) -> str:
    """Retrieve active conversation thread_id for a user or create a new one."""
    uid = str(user_id)
    try:
        r = _get_redis()
        tid = r.get(f"{REDIS_PREFIX}:user:{uid}:active_thread")
        if tid:
            return tid
    except Exception as e:
        print(f"[WARN] Redis get_user_active_thread failed: {e}")
    new_tid = str(uuid.uuid4())
    set_user_active_thread(uid, new_tid)
    return new_tid


def set_user_active_thread(user_id: str | int, thread_id: str):
    """Set active conversation thread_id for a user."""
    uid = str(user_id)
    try:
        r = _get_redis()
        r.set(f"{REDIS_PREFIX}:user:{uid}:active_thread", thread_id)
    except Exception as e:
        print(f"[WARN] Redis set_user_active_thread failed: {e}")


def add_user_collection(user_id: str | int, collection_id: str):
    """Associate a collection with a specific user."""
    uid = str(user_id)
    try:
        r = _get_redis()
        r.sadd(f"{REDIS_PREFIX}:user:{uid}:projects", collection_id)
    except Exception as e:
        print(f"[WARN] Redis add_user_collection failed: {e}")


def remove_user_collection(user_id: str | int, collection_id: str):
    """Disassociate a collection from a specific user and clear active pointer if matched."""
    uid = str(user_id)
    try:
        r = _get_redis()
        r.srem(f"{REDIS_PREFIX}:user:{uid}:projects", collection_id)
        active = r.get(f"{REDIS_PREFIX}:user:{uid}:active_project")
        if active == collection_id:
            r.delete(f"{REDIS_PREFIX}:user:{uid}:active_project")
    except Exception as e:
        print(f"[WARN] Redis remove_user_collection failed: {e}")


def list_user_collections(user_id: str | int) -> List[Dict[str, Any]]:
    """List all collections created by or assigned to a specific user."""
    uid = str(user_id)
    try:
        r = _get_redis()
        collection_ids = r.smembers(f"{REDIS_PREFIX}:user:{uid}:projects")
        if not collection_ids:
            # Auto-link existing collections in system so the user immediately gets access
            all_cids = r.smembers(_collections_set_key())
            for cid in all_cids:
                if get_collection(cid):
                    r.sadd(f"{REDIS_PREFIX}:user:{uid}:projects", cid)
            collection_ids = r.smembers(f"{REDIS_PREFIX}:user:{uid}:projects")

        collections = []
        for cid in collection_ids:
            coll = get_collection(cid)
            if coll:
                conv_count = r.scard(_collection_conversations_key(cid))
                coll["conversation_count"] = conv_count
                collections.append(coll)
            else:
                r.srem(f"{REDIS_PREFIX}:user:{uid}:projects", cid)
        collections.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return collections
    except Exception as e:
        print(f"[WARN] Redis list_user_collections failed: {e}")
        return []


# Backward-compat aliases for Telegram helpers
get_user_active_project = get_user_active_collection
set_user_active_project = set_user_active_collection
add_user_project = add_user_collection
remove_user_project = remove_user_collection
list_user_projects = list_user_collections


def get_user_resume(user_id: str | int) -> Dict[str, Any]:
    """Retrieve user's active resume details (text, file path, filename)."""
    uid = str(user_id)
    try:
        r = _get_redis()
        raw = r.get(f"{REDIS_PREFIX}:user:{uid}:resume")
        if raw:
            return json.loads(raw)
    except Exception as e:
        print(f"[WARN] Redis get_user_resume failed: {e}")
    return {}


def set_user_resume(user_id: str | int, resume_data: Dict[str, Any]):
    """Save user's active resume details (text, file path, filename)."""
    uid = str(user_id)
    try:
        r = _get_redis()
        r.set(f"{REDIS_PREFIX}:user:{uid}:resume", json.dumps(resume_data))
    except Exception as e:
        print(f"[WARN] Redis set_user_resume failed: {e}")


def clear_user_resume(user_id: str | int):
    """Clear user's attached resume."""
    uid = str(user_id)
    try:
        r = _get_redis()
        r.delete(f"{REDIS_PREFIX}:user:{uid}:resume")
    except Exception as e:
        print(f"[WARN] Redis clear_user_resume failed: {e}")


def get_user_email_creds(user_id: str | int) -> Dict[str, str]:
    """Retrieve user's personal Gmail credentials for direct dispatch."""
    uid = str(user_id)
    try:
        r = _get_redis()
        raw = r.get(f"{REDIS_PREFIX}:user:{uid}:email_creds")
        if raw:
            return json.loads(raw)
    except Exception as e:
        print(f"[WARN] Redis get_user_email_creds failed: {e}")
    return {}


def set_user_email_creds(user_id: str | int, email: str, app_password: str):
    """Save user's personal Gmail credentials for direct dispatch."""
    uid = str(user_id)
    try:
        r = _get_redis()
        r.set(
            f"{REDIS_PREFIX}:user:{uid}:email_creds",
            json.dumps({"email": email.strip(), "app_password": app_password.strip()}),
        )
    except Exception as e:
        print(f"[WARN] Redis set_user_email_creds failed: {e}")
