"""
LinkedIn AI Career Strategist — Streamlit Web Application.
- Sidebar: Multi-Collection Management & Collection-Scoped Conversations.
- Exa.ai Agent: Automated job data collection & deduplicated storage in collection Qdrant vector stores.
- Near Input: One-click "Collect New Jobs" to append fresh jobs to active collection.
- Search & Auto-Apply: Powered by LangGraph, Qdrant Hybrid Retrieval, smtplib email dispatch.
"""
import streamlit as st
import uuid
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

from resume_parser import parse_resume_file
from agent_graph import (
    stream_assistant_turn,
    DEFAULT_MODEL,
    intent_detection_node,
)
from crew_workflow import (
    classify_intent_with_router_agent,
    run_company_intelligence_crew,
    run_job_application_crew,
    run_general_career_crew,
)
from redis_store import (
    get_session_state,
    save_session_state,
    list_conversations,
    delete_session,
    clear_all_conversations,
    create_collection,
    get_collection,
    list_collections,
    delete_collection,
    update_collection,
)
from nodes.exa_node import collect_jobs_via_exa, web_search_company
from nodes.store_node import store_jobs_to_collection
from email_sender import send_application_email
import ollama

# ------------------------------------------------------------
# Page Configuration & Modern Styling
# ------------------------------------------------------------
st.set_page_config(
    page_title="LinkedIn AI Career Strategist",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* Main Layout Styling */
    .main-title {
        font-size: 2.1rem;
        font-weight: 700;
        color: #0A66C2;
        margin-bottom: 0.1rem;
    }
    .sub-title {
        font-size: 0.98rem;
        color: #555;
        margin-bottom: 1.2rem;
    }
    .collection-pill {
        display: inline-flex;
        align-items: center;
        background-color: #e8f4fd;
        color: #0A66C2;
        border: 1px solid #b2d7f7;
        padding: 0.3rem 0.8rem;
        border-radius: 16px;
        font-size: 0.9rem;
        font-weight: 600;
        margin-bottom: 0.6rem;
    }
    .welcome-card {
        padding: 1.5rem;
        border-radius: 12px;
        # background: linear-gradient(135deg, #f0f7ff 0%, #ffffff 100%);
        border: 1px solid #cce4f7;
        margin-bottom: 1.5rem;
    }
    .welcome-card h3 {
        color: #0A66C2;
        margin-top: 0;
        font-size: 1.3rem;
    }
    .resume-badge {
        display: inline-flex;
        align-items: center;
        background-color: #e8f4fd;
        color: #0A66C2;
        border: 1px solid #b2d7f7;
        padding: 0.35rem 0.75rem;
        border-radius: 20px;
        font-size: 0.88rem;
        font-weight: 500;
    }
    .job-preview-card {
        padding: 0.75rem 1rem;
        border-radius: 8px;
        background-color: #f8f9fa;
        border-left: 4px solid #0A66C2;
        margin-bottom: 0.5rem;
        font-size: 0.92rem;
    }
    .email-card {
        padding: 1.2rem 1.4rem;
        border-radius: 12px;
        background-color: #ffffff;
        border: 2px solid #0A66C2;
        box-shadow: 0 4px 12px rgba(10, 102, 194, 0.08);
        margin-top: 1rem;
        margin-bottom: 1.2rem;
    }
    .email-card h4 {
        color: #0A66C2;
        margin-top: 0;
        margin-bottom: 0.5rem;
    }
    .help-box {
        background-color: #f6f8fa;
        border: 1px solid #e1e4e8;
        border-radius: 8px;
        padding: 0.85rem;
        font-size: 0.88rem;
        line-height: 1.5;
        margin-bottom: 0.8rem;
    }
    .stButton > button {
        border-radius: 8px;
    }
    div[data-testid="stSidebar"] hr {
        margin-top: 0.75rem;
        margin-bottom: 0.75rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------
# Session State Initialization: Collections & Conversations
# ------------------------------------------------------------
collections = list_collections()

if "collection_id" not in st.session_state or not st.session_state["collection_id"]:
    if collections:
        st.session_state["collection_id"] = collections[0].get("collection_id") or collections[0].get("project_id")
    else:
        # Create initial default collection
        init_cid = create_collection(
            name="General Software Roles",
            search_query="Find active python and software engineering jobs in India",
            qdrant_collection="linkedin_jobs",
        )
        st.session_state["collection_id"] = init_cid
        collections = list_collections()

active_collection_id = st.session_state.get("collection_id")
active_collection = get_collection(active_collection_id) if active_collection_id else None

# Session State Initialization: Active Conversation
if "thread_id" not in st.session_state:
    c_convs = list_conversations(collection_id=active_collection_id)
    if c_convs:
        st.session_state["thread_id"] = c_convs[0]["thread_id"]
    else:
        st.session_state["thread_id"] = str(uuid.uuid4())

current_thread_id = st.session_state["thread_id"]
current_session = get_session_state(current_thread_id)

# ------------------------------------------------------------
# Sidebar: Collection Management & Collection-Scoped Conversations
# ------------------------------------------------------------
with st.sidebar:
    st.markdown("### 📁 Collections")

    # Collection Selector Dropdown
    if collections:
        collection_map = {c["collection_id"]: f"📁 {c['name']} ({c.get('total_jobs', 0)} jobs)" for c in collections}
        selected_cid = st.selectbox(
            "Select Active Collection",
            options=list(collection_map.keys()),
            format_func=lambda x: collection_map.get(x, x),
            index=list(collection_map.keys()).index(active_collection_id) if active_collection_id in collection_map else 0,
            label_visibility="collapsed",
        )
        if selected_cid != active_collection_id:
            st.session_state["collection_id"] = selected_cid
            rem_convs = list_conversations(collection_id=selected_cid)
            st.session_state["thread_id"] = rem_convs[0]["thread_id"] if rem_convs else str(uuid.uuid4())
            st.rerun()

    # "+ Create New Collection" Expander
    with st.expander("➕ Create New Collection", expanded=False):
        with st.form("new_collection_query_form", clear_on_submit=False):
            p_query = st.text_area(
                "What jobs or data would you like to collect?*",
                placeholder="e.g. 'Find active Senior Python developer jobs in Pune' or 'Tell me about hiring at Google' or 'Apply to Microsoft for Machine Learning Engineer role'...",
                height=100,
                help="Describe what jobs to collect, ask about a company's hiring/details, or request to apply for a role.",
            )
            p_name_input = st.text_input(
                "Collection Name (Optional):",
                placeholder="e.g. Pune Backend Roles (Auto-generated if blank)",
            )
            p_resume = st.file_uploader(
                "Candidate Resume (Optional):",
                type=["pdf", "png", "jpg", "jpeg", "webp"],
                key="create_coll_resume",
                help="Attach candidate profile for targeted job collection, company matching, or auto-apply drafting",
            )

            submitted = st.form_submit_button("🚀 Submit Query / Create Collection", type="primary", use_container_width=True)
            if submitted:
                if not p_query.strip():
                    st.error("Please enter a query or job requirements.")
                else:
                    # 1. Parse Resume if provided
                    p_resume_text = None
                    p_resume_filename = None
                    p_resume_filepath = None
                    if p_resume is not None:
                        file_bytes = p_resume.read()
                        p_resume_filename = p_resume.name
                        p_resume_text, _ = parse_resume_file(file_bytes, p_resume.name)
                        os.makedirs("uploads", exist_ok=True)
                        clean_fn = "".join(c for c in p_resume.name if c.isalnum() or c in "._-")
                        save_path = os.path.join("uploads", f"proj_{uuid.uuid4().hex[:6]}_{clean_fn}")
                        with open(save_path, "wb") as f:
                            f.write(file_bytes)
                        p_resume_filepath = save_path

                    # 2. Intent Detection via CrewAI Intake & Routing Coordinator Agent
                    with st.spinner("🎯 Intake Coordinator Agent is analyzing your query intent..."):
                        detected_intent = classify_intent_with_router_agent(
                            user_query=p_query.strip(),
                            model_name=DEFAULT_MODEL,
                        )

                    # Collection Name resolution
                    if p_name_input.strip():
                        coll_name_label = p_name_input.strip()
                    else:
                        clean_q = p_query.strip().split("\n")[0]
                        if detected_intent == "job_search":
                            coll_name_label = f"Jobs: {clean_q}"
                        elif detected_intent == "web_search":
                            coll_name_label = f"Intel: {clean_q}"
                        elif detected_intent == "apply_job":
                            coll_name_label = f"Apply: {clean_q}"
                        else:
                            coll_name_label = clean_q

                    new_tid = str(uuid.uuid4())

                    # 3. Process according to LLM Intent
                    if detected_intent == "job_search":
                        with st.spinner("Exa.ai research agent is discovering and structuring active jobs..."):
                            jobs, exa_metrics = collect_jobs_via_exa(
                                query=p_query.strip(),
                                resume_text=p_resume_text,
                            )
                            new_cid = create_collection(
                                name=coll_name_label,
                                search_query=p_query.strip(),
                                exa_metrics=exa_metrics,
                            )
                            coll_data = get_collection(new_cid)
                            qdrant_coll = coll_data.get("qdrant_collection") or coll_data.get("collection_name")
                            store_stats = store_jobs_to_collection(jobs, collection_name=qdrant_coll)
                            update_collection(new_cid, {
                                "total_jobs": store_stats["total_jobs"],
                                "exa_metrics": exa_metrics,
                            })

                            initial_msg = (
                                f"✅ **Collected {len(jobs)} active jobs** via Exa.ai agent for: *\"{p_query.strip()}\"*!\n\n"
                                f"- **Stored in Database**: {store_stats['stored_count']} non-duplicate jobs\n"
                                f"- **Collection**: `{qdrant_coll}`\n\n"
                                f"All jobs conform strictly to the 35 schema fields with verified statuses and complete descriptions.\n\n"
                                f"💡 **Next Step**: You can ask:\n"
                                f"- *\"Show top 5 recommended jobs for my profile\"*\n"
                                f"- *\"Compare these jobs with my resume\"*\n"
                                f"- *\"Apply to job 1\"*"
                            )
                            save_session_state(new_tid, {
                                "messages": [
                                    {"role": "user", "content": p_query.strip()},
                                    {"role": "assistant", "content": initial_msg},
                                ],
                                "current_jobs": jobs[:5],
                                "resume_text": p_resume_text,
                                "resume_filename": p_resume_filename,
                                "resume_file_path": p_resume_filepath,
                                "email_draft": None,
                                "history_summary": None,
                                "title": p_query.strip(),
                                "collection_id": new_cid,
                            }, collection_id=new_cid)

                            st.session_state["collection_id"] = new_cid
                            st.session_state["thread_id"] = new_tid
                            st.success(f"✅ Created collection '{coll_name_label}' with {store_stats['stored_count']} jobs!")
                            st.balloons()
                            st.rerun()

                    elif detected_intent == "web_search":
                        with st.spinner(f"🌐 Company Intelligence Agent researching: '{p_query.strip()[:40]}' via Exa..."):
                            synthesis = run_company_intelligence_crew(
                                user_query=p_query.strip(),
                                model_name=DEFAULT_MODEL,
                            )

                            new_cid = create_collection(
                                name=coll_name_label,
                                search_query=p_query.strip(),
                            )
                            save_session_state(new_tid, {
                                "messages": [
                                    {"role": "user", "content": p_query.strip()},
                                    {"role": "assistant", "content": synthesis},
                                ],
                                "current_jobs": [],
                                "resume_text": p_resume_text,
                                "resume_filename": p_resume_filename,
                                "resume_file_path": p_resume_filepath,
                                "email_draft": None,
                                "history_summary": None,
                                "title": p_query.strip(),
                                "collection_id": new_cid,
                            }, collection_id=new_cid)

                            st.session_state["collection_id"] = new_cid
                            st.session_state["thread_id"] = new_tid
                            st.success(f"✅ Created collection '{coll_name_label}' with Company Research!")
                            st.rerun()

                    elif detected_intent == "apply_job":
                        with st.spinner("📝 Application Specialist Agent drafting personalized application & cover letter..."):
                            app_res = run_job_application_crew(
                                user_query=p_query.strip(),
                                resume_text=p_resume_text,
                                model_name=DEFAULT_MODEL,
                            )
                            draft = app_res.get("email_draft")
                            assistant_msg = app_res.get("final_text") or f"Prepared application email for {draft.get('job_title', 'Role')} at {draft.get('company', 'Company')}."

                            new_cid = create_collection(
                                name=coll_name_label,
                                search_query=p_query.strip(),
                            )
                            save_session_state(new_tid, {
                                "messages": [
                                    {"role": "user", "content": p_query.strip()},
                                    {"role": "assistant", "content": assistant_msg},
                                ],
                                "current_jobs": [],
                                "resume_text": p_resume_text,
                                "resume_filename": p_resume_filename,
                                "resume_file_path": p_resume_filepath,
                                "email_draft": draft,
                                "history_summary": None,
                                "title": p_query.strip(),
                                "collection_id": new_cid,
                            }, collection_id=new_cid)

                            st.session_state["collection_id"] = new_cid
                            st.session_state["thread_id"] = new_tid
                            st.success(f"✅ Created collection '{coll_name_label}' with Application Draft!")
                            st.rerun()

                    else: # general
                        with st.spinner("🤝 Career Advisor Agent is formulating recommendations..."):
                            assistant_msg = run_general_career_crew(
                                user_query=p_query.strip(),
                                resume_text=p_resume_text,
                                model_name=DEFAULT_MODEL,
                            )
                            new_cid = create_collection(
                                name=coll_name_label,
                                search_query=p_query.strip(),
                            )
                            save_session_state(new_tid, {
                                "messages": [
                                    {"role": "user", "content": p_query.strip()},
                                    {"role": "assistant", "content": assistant_msg},
                                ],
                                "current_jobs": [],
                                "resume_text": p_resume_text,
                                "resume_filename": p_resume_filename,
                                "resume_file_path": p_resume_filepath,
                                "email_draft": None,
                                "history_summary": None,
                                "title": p_query.strip(),
                                "collection_id": new_cid,
                            }, collection_id=new_cid)

                            st.session_state["collection_id"] = new_cid
                            st.session_state["thread_id"] = new_tid
                            st.success(f"✅ Created collection '{coll_name_label}'!")
                            st.rerun()

    # Active Collection Exa Metrics & Actions
    if active_collection:
        metrics = active_collection.get("exa_metrics") or {}
        if metrics:
            with st.expander(f"⚡ Exa Metrics ({active_collection.get('total_jobs', 0)} jobs)", expanded=False):
                st.caption(f"**Run ID**: `{metrics.get('run_id') or 'N/A'}`")
                st.caption(f"**Execution Time**: {metrics.get('duration_seconds', 0)}s")
                st.caption(f"**Cost**: ${metrics.get('cost_dollars', 0.0):.4f}")
                st.caption(f"**Qdrant Store**: `{active_collection.get('qdrant_collection') or active_collection.get('collection_name')}`")

        if st.button("🗑️ Delete Active Collection", key="delete_active_coll_btn", help="Delete this collection and its Qdrant vector store"):
            delete_collection(active_collection_id)
            rem = list_collections()
            st.session_state["collection_id"] = rem[0]["collection_id"] if rem else None
            st.rerun()

    st.markdown("---")

    # Collection-Scoped Conversations
    coll_display_name = active_collection.get("name", "Collection") if active_collection else "Conversations"
    st.markdown(f"### 💬 Conversations")
    st.caption(f"Linked to: **{coll_display_name}**")

    # "+ New Chat" button for active project
    if st.button("➕ New Chat", use_container_width=True, type="primary"):
        new_id = str(uuid.uuid4())
        st.session_state["thread_id"] = new_id
        save_session_state(new_id, {
            "messages": [],
            "current_jobs": [],
            "resume_text": None,
            "resume_filename": None,
            "resume_file_path": None,
            "email_draft": None,
            "history_summary": None,
            "title": "New Chat",
            "collection_id": active_collection_id,
        }, collection_id=active_collection_id)
        st.rerun()

    # List conversations for active project
    conversations = list_conversations(collection_id=active_collection_id)

    if not conversations:
        st.caption("No conversations in this collection yet. Start chatting!")
    else:
        for conv in conversations:
            tid = conv["thread_id"]
            title = conv.get("title", "New Chat")
            is_active = (tid == current_thread_id)
            msg_count = conv.get("message_count", 0)

            full_title = conv.get("full_title", title)
            col_btn, col_del = st.columns([0.83, 0.17])
            with col_btn:
                btn_prefix = "👉 " if is_active else "💬 "
                btn_type = "primary" if is_active else "secondary"
                label = f"{btn_prefix}{title}"
                if st.button(
                    label,
                    key=f"conv_btn_{tid}",
                    use_container_width=True,
                    type=btn_type,
                    help=f"{full_title} ({msg_count} messages)",
                ):
                    st.session_state["thread_id"] = tid
                    st.rerun()

            with col_del:
                if st.button("🗑️", key=f"del_btn_{tid}", help=f"Delete '{full_title}'"):
                    delete_session(tid)
                    if st.session_state["thread_id"] == tid:
                        remaining = list_conversations(collection_id=active_collection_id)
                        st.session_state["thread_id"] = remaining[0]["thread_id"] if remaining else str(uuid.uuid4())
                    st.rerun()

        st.markdown("---")
        if st.button("🧹 Clear Collection Chats", use_container_width=True):
            clear_all_conversations(collection_id=active_collection_id)
            st.session_state["thread_id"] = str(uuid.uuid4())
            st.rerun()


# ------------------------------------------------------------
# Main Chat Area
# ------------------------------------------------------------
st.markdown('<div class="main-title">💼 LinkedIn AI Job & Career Strategist</div>', unsafe_allow_html=True)
if active_collection:
    st.markdown(
        f'<div class="collection-pill">📁 Active Collection: {active_collection.get("name")} ({active_collection.get("total_jobs", 0)} Jobs in DB)</div>',
        unsafe_allow_html=True,
    )
st.markdown(
    f'<div class="sub-title">Powered by <b>Ollama</b> (<code>{DEFAULT_MODEL}</code>) • CrewAI • Exa.ai Intelligence • Qdrant Isolation</div>',
    unsafe_allow_html=True,
)

# Active jobs expander if current conversation has cached jobs
active_jobs = current_session.get("current_jobs", [])
if active_jobs:
    with st.expander(f"📋 View Active Jobs in Memory ({len(active_jobs)} jobs retrieved)", expanded=False):
        for j in active_jobs:
            st.markdown(
                f"""
                <div class="job-preview-card">
                    <b>#{j.get('rank', '')} {j.get('jobTitle') or j.get('Job Title', 'Job')}</b> at <b>{j.get('company') or j.get('Company', 'N/A')}</b><br>
                    📍 <i>{j.get('city') or j.get('Location', 'N/A')}</i> | 🏢 {j.get('experienceLevel', 'N/A')} | 💰 {j.get('salaryMin', 'N/A')} - {j.get('salaryMax', 'N/A')}<br>
                    🔑 <b>Skills</b>: {j.get('skillsRequired', 'N/A')}<br>
                    📧 <b>Email</b>: <code>{j.get('companyEmail') or j.get('Company Email') or 'N/A'}</code> | 🔗 <a href="{j.get('sourceUrl') or j.get('Source URL', '#')}" target="_blank">Apply Link</a>
                </div>
                """,
                unsafe_allow_html=True,
            )

# Render Chat History
messages = current_session.get("messages", [])

if not messages:
    st.markdown(
        f"""
        <div class="welcome-card">
            <h3>👋 Welcome to your LinkedIn AI Career Assistant!</h3>
            <p>You are currently in collection <b>{active_collection.get('name') if active_collection else 'General'}</b>.</p>
            <ul>
                <li>🔍 <b>Find Jobs</b>: Ask for roles (e.g., <i>"Find remote Python developer jobs"</i>). Retrieves strictly from this project's database.</li>
                <li>🌐 <b>Company Web Search</b>: Ask <i>"Is Google hiring for AI engineers?"</i> or <i>"Tell me about Carbon Robotics"</i>.</li>
                <li>📎 <b>Attach Resume</b>: Upload your PDF or Image resume below to receive personalized fit ratings and <b>Skill Gap Analysis</b>.</li>
                <li>🚀 <b>Auto Apply</b>: Say <i>"Apply to job 1"</i> or <i>"Apply to Microsoft"</i> to draft and send a tailored application email.</li>
                <li>🔄 <b>Append New Jobs</b>: Click <b>"Collect New Jobs"</b> near the input bar to fetch fresh jobs using Exa.ai without duplicates!</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        with st.chat_message(role):
            st.markdown(content)


# ------------------------------------------------------------
# Application Email Dispatch Interface (When Draft is Ready)
# ------------------------------------------------------------
email_draft = current_session.get("email_draft")

if email_draft and email_draft.get("status") == "ready_to_send":
    with st.container():
        st.markdown(
            f"""
            <div class="email-card">
                <h4>✉️ Send Application Email: {email_draft.get('job_title')} at {email_draft.get('company')}</h4>
                <p style="color: #555; margin-bottom: 0.5rem;">Review the email details and choose your preferred sending method below:</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander("📝 Review / Edit Application Draft & Recipient", expanded=True):
            col_to, col_sub = st.columns([0.45, 0.55])
            with col_to:
                target_recipient = st.text_input(
                    "Company Recipient Email",
                    value=email_draft.get("recipient_email", ""),
                    key="draft_recipient_input",
                )
            with col_sub:
                target_subject = st.text_input(
                    "Email Subject Line",
                    value=email_draft.get("subject", ""),
                    key="draft_subject_input",
                )

            target_body = st.text_area(
                "Cover Letter / Email Body",
                value=email_draft.get("body", ""),
                height=220,
                key="draft_body_input",
            )

            resume_file_display = current_session.get("resume_filename") or "Candidate_Resume.txt"
            st.caption(f"📎 **Attached Resume File**: `{resume_file_display}`")

        # 2 Dispatch Options
        tab_app_mail, tab_user_gmail = st.tabs([
            "🚀 Option 1: Send via App Mail (apply@ourapp.com)",
            "🔐 Option 2: Send from Your Personal Gmail",
        ])

        # --- Option 1: App Mail ---
        with tab_app_mail:
            st.info(
                "**How it works**: We send the email from our verified server (`apply@ourapp.com`). "
                "Your email address is placed in the **Reply-To** header, so the company's recruiter replies directly to your inbox."
            )
            app_user_email = st.text_input(
                "Enter Your Email Address (for recruiter reply):",
                placeholder="you@example.com",
                key="opt1_user_email",
            )

            col_send1, col_dismiss1 = st.columns([0.35, 0.65])
            with col_send1:
                if st.button("🚀 Send via App Mail", key="btn_send_app_mail", type="primary", use_container_width=True):
                    if not app_user_email or "@" not in app_user_email:
                        st.error("Please enter a valid email address for Reply-To.")
                    elif not target_recipient or "@" not in target_recipient:
                        st.error("Please enter a valid company recipient email.")
                    else:
                        with st.spinner("Dispatching application email via smtplib..."):
                            res = send_application_email(
                                recipient_email=target_recipient.strip(),
                                subject=target_subject.strip(),
                                body=target_body.strip(),
                                sender_option="app",
                                user_email=app_user_email.strip(),
                                resume_path=current_session.get("resume_file_path"),
                                resume_text=current_session.get("resume_text"),
                                resume_filename=current_session.get("resume_filename"),
                            )
                            if res["success"]:
                                st.success(res["message"])
                                current_session["messages"].append({
                                    "role": "assistant",
                                    "content": f"🎉 **Application Sent!**\n\n{res['message']}\n\n**To**: `{target_recipient}`\n**Subject**: `{target_subject}`"
                                })
                                email_draft["status"] = "sent"
                                current_session["email_draft"] = email_draft
                                save_session_state(current_thread_id, current_session, collection_id=active_collection_id)
                                st.balloons()
                                st.rerun()
                            else:
                                st.error(res["message"])

            with col_dismiss1:
                if st.button("❌ Dismiss Draft", key="btn_dismiss_draft_1"):
                    current_session["email_draft"] = None
                    save_session_state(current_thread_id, current_session, collection_id=active_collection_id)
                    st.rerun()

        # --- Option 2: User's Personal Gmail ---
        with tab_user_gmail:
            st.markdown(
                """
                <div class="help-box">
                    <b>🔑 How to Generate a 16-Digit Google App Password:</b><br>
                    1. Visit <a href="https://myaccount.google.com/apppasswords" target="_blank"><b>myaccount.google.com/apppasswords</b></a> in your browser.<br>
                    2. Sign in with your Google account <i>(Make sure 2-Step Verification is ON)</i>.<br>
                    3. Under <b>"App name"</b>, enter <code>LinkedIn Assistant</code> and click <b>Create</b>.<br>
                    4. Google will display a 16-letter password (e.g., <code>abcd efgh ijkl mnop</code>).<br>
                    5. Copy that 16-digit code and paste it below. <i>(Your actual Google account password is never required or stored)</i>.
                </div>
                """,
                unsafe_allow_html=True,
            )

            col_gm_user, col_gm_pass = st.columns(2)
            with col_gm_user:
                gmail_user_email = st.text_input(
                    "Your Gmail Address:",
                    placeholder="yourname@gmail.com",
                    key="opt2_gmail_user",
                )
            with col_gm_pass:
                gmail_app_pass = st.text_input(
                    "16-Digit Google App Password:",
                    type="password",
                    placeholder="abcd efgh ijkl mnop",
                    key="opt2_gmail_pass",
                    help="Generated from myaccount.google.com/apppasswords",
                )

            col_send2, col_dismiss2 = st.columns([0.35, 0.65])
            with col_send2:
                if st.button("🚀 Send from My Gmail", key="btn_send_user_gmail", type="primary", use_container_width=True):
                    if not gmail_user_email or "@" not in gmail_user_email:
                        st.error("Please enter a valid Gmail address.")
                    elif not gmail_app_pass or len(gmail_app_pass.replace(" ", "")) < 12:
                        st.error("Please enter your 16-digit Google App Password.")
                    elif not target_recipient or "@" not in target_recipient:
                        st.error("Please enter a valid company recipient email.")
                    else:
                        with st.spinner("Connecting to Gmail SMTP server & sending email..."):
                            res = send_application_email(
                                recipient_email=target_recipient.strip(),
                                subject=target_subject.strip(),
                                body=target_body.strip(),
                                sender_option="user_gmail",
                                user_email=gmail_user_email.strip(),
                                user_app_password=gmail_app_pass.strip(),
                                resume_path=current_session.get("resume_file_path"),
                                resume_text=current_session.get("resume_text"),
                                resume_filename=current_session.get("resume_filename"),
                            )
                            if res["success"]:
                                st.success(res["message"])
                                current_session["messages"].append({
                                    "role": "assistant",
                                    "content": f"🎉 **Application Sent from your Gmail!**\n\n{res['message']}\n\n**To**: `{target_recipient}`\n**Subject**: `{target_subject}`"
                                })
                                email_draft["status"] = "sent"
                                current_session["email_draft"] = email_draft
                                save_session_state(current_thread_id, current_session, collection_id=active_collection_id)
                                st.balloons()
                                st.rerun()
                            else:
                                st.error(res["message"])

            with col_dismiss2:
                if st.button("❌ Dismiss Draft", key="btn_dismiss_draft_2"):
                    current_session["email_draft"] = None
                    save_session_state(current_thread_id, current_session, collection_id=active_collection_id)
                    st.rerun()

        st.markdown("---")


# ------------------------------------------------------------
# Near Input: One-Click Collect New Jobs & Resume Attachment
# ------------------------------------------------------------
attached_resume_text = current_session.get("resume_text")
attached_resume_filename = current_session.get("resume_filename")

with st.container():
    col_fetch_jobs, col_resume_mgr = st.columns([0.48, 0.52])

    with col_fetch_jobs:
        if st.button("🔄 Collect New Jobs (Exa Agent)", use_container_width=True, help="Call Exa.ai agent with original collection criteria and append non-duplicate jobs to database"):
            if active_collection:
                with st.spinner(f"Exa agent finding new jobs for '{active_collection.get('name')}'..."):
                    new_jobs, new_metrics = collect_jobs_via_exa(
                        query=active_collection.get("search_query") or active_collection.get("name"),
                        resume_text=attached_resume_text,
                        filters=active_collection.get("search_metadata"),
                    )
                    qdrant_name = active_collection.get("qdrant_collection") or active_collection.get("collection_name", "linkedin_jobs")
                    store_res = store_jobs_to_collection(
                        new_jobs,
                        collection_name=qdrant_name,
                    )
                    update_collection(active_collection_id, {
                        "total_jobs": store_res["total_jobs"],
                        "exa_metrics": new_metrics,
                    })
                    st.success(f"✅ Added {store_res['stored_count']} new jobs ({store_res['skipped_count']} duplicate jobs skipped). Total in collection: {store_res['total_jobs']}")
                    st.rerun()
            else:
                st.warning("Please select or create a collection first.")

    with col_resume_mgr:
        if attached_resume_text:
            col_rbadge, col_rdetach = st.columns([0.75, 0.25])
            with col_rbadge:
                rname = attached_resume_filename or "Resume Attached"
                st.markdown(
                    f'<div class="resume-badge">📄 Resume: <b>{rname}</b></div>',
                    unsafe_allow_html=True,
                )
            with col_rdetach:
                if st.button("❌ Detach", key="detach_resume_btn", help="Remove resume from this conversation"):
                    old_path = current_session.get("resume_file_path")
                    if old_path and os.path.exists(old_path):
                        try:
                            os.remove(old_path)
                        except Exception:
                            pass
                    current_session["resume_text"] = None
                    current_session["resume_filename"] = None
                    current_session["resume_file_path"] = None
                    save_session_state(current_thread_id, current_session, collection_id=active_collection_id)
                    st.rerun()
        else:
            with st.expander("📎 Attach Resume (PDF/Image)", expanded=False):
                uploaded_file = st.file_uploader(
                    "Upload Resume",
                    type=["pdf", "png", "jpg", "jpeg", "webp"],
                    key=f"resume_uploader_{current_thread_id}",
                    label_visibility="collapsed",
                )
                if uploaded_file is not None:
                    with st.spinner("Extracting resume..."):
                        fbytes = uploaded_file.read()
                        extracted_text, _ = parse_resume_file(fbytes, uploaded_file.name)
                        if extracted_text:
                            os.makedirs("uploads", exist_ok=True)
                            clean_fn = "".join(c for c in uploaded_file.name if c.isalnum() or c in "._-")
                            save_path = os.path.join("uploads", f"{current_thread_id[:8]}_{clean_fn}")
                            with open(save_path, "wb") as f:
                                f.write(fbytes)

                            current_session["resume_text"] = extracted_text
                            current_session["resume_filename"] = uploaded_file.name
                            current_session["resume_file_path"] = save_path
                            save_session_state(current_thread_id, current_session, collection_id=active_collection_id)
                            st.success(f"✅ Attached: {uploaded_file.name}")
                            st.rerun()
                        else:
                            st.error("Could not extract text from file.")


# ------------------------------------------------------------
# Chat Input & Streaming Execution
# ------------------------------------------------------------
user_query = st.chat_input("Ask for jobs, company intelligence ('Is Google hiring?'), or say 'Apply to job 1'...")

if user_query:
    # 1. Display User Message immediately
    with st.chat_message("user"):
        st.markdown(user_query)
        if attached_resume_filename:
            st.caption(f"📎 Attached: `{attached_resume_filename}`")

    # 2. Assistant Response Container
    with st.chat_message("assistant"):
        status_placeholder = st.empty()

        def on_status(msg: str):
            status_placeholder.status(msg, state="running")

        with st.spinner("Thinking via Ollama..."):
            token_generator = stream_assistant_turn(
                user_query=user_query,
                resume_text=attached_resume_text,
                thread_id=current_thread_id,
                collection_id=active_collection_id,
                model_name=DEFAULT_MODEL,
                status_callback=on_status,
                chat_history=messages,
            )
            full_response = st.write_stream(token_generator)
            status_placeholder.empty()

    # 3. Rerun to refresh conversation title and state from Redis
    st.rerun()
