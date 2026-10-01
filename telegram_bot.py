"""
Telegram Bot Interface for LinkedIn AI Career Assistant.
Provides:
1. Multi-Collection Management (Create, Switch, Delete, Browse)
2. Interactive Exa.ai Job Gathering with real-time feedback
3. PDF/Image Resume Ingestion via Telegram Documents & Photos
4. Autonomous Qdrant Search & LLM Career Strategy Guidance
5. Direct 1-Click Application Email Dispatch (App Mail & Personal Gmail)
"""
from __future__ import annotations

import asyncio
import html
import logging
import os
import uuid
from typing import Dict, Any, List, Optional

from dotenv import load_dotenv
from telegram import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Update,
    constants,
)
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    ConversationHandler,
    MessageHandler,
    filters,
)

# Load environment variables
load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()

# Internal pipeline modules
from agent_graph import stream_assistant_turn
from nodes.exa_node import collect_jobs_via_exa
from nodes.store_node import store_jobs_to_collection
from resume_parser import parse_resume_file
from email_sender import send_application_email
from redis_store import (
    create_collection,
    get_collection,
    delete_collection,
    update_collection,
    get_session_state,
    save_session_state,
    get_user_active_collection,
    set_user_active_collection,
    get_user_active_thread,
    set_user_active_thread,
    add_user_collection,
    remove_user_collection,
    list_user_collections,
    get_user_resume,
    set_user_resume,
    clear_user_resume,
    get_user_email_creds,
    set_user_email_creds,
)

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("telegram_bot")

# ConversationHandler states for New Collection wizard
WAITING_COLLECTION_NAME, WAITING_COLLECTION_CRITERIA = range(2)
WAITING_PROJECT_NAME, WAITING_PROJECT_CRITERIA = WAITING_COLLECTION_NAME, WAITING_COLLECTION_CRITERIA


# ============================================================
# Helper Functions
# ============================================================

def split_message(text: str, max_len: int = 4000) -> List[str]:
    """
    Split long text into Telegram-compliant chunks (< 4096 characters),
    splitting at double-newlines, single-newlines, or space boundaries.
    """
    if len(text) <= max_len:
        return [text]

    chunks = []
    while text:
        if len(text) <= max_len:
            chunks.append(text)
            break

        # Try splitting at double newline
        split_idx = text.rfind("\n\n", 0, max_len)
        if split_idx == -1:
            # Try splitting at single newline
            split_idx = text.rfind("\n", 0, max_len)
        if split_idx == -1:
            # Try splitting at space
            split_idx = text.rfind(" ", 0, max_len)
        if split_idx == -1:
            # Fallback: hard slice
            split_idx = max_len

        chunk = text[:split_idx].strip()
        if chunk:
            chunks.append(chunk)
        text = text[split_idx:].strip()

    return chunks


async def send_typing_action(chat_id: int, context: ContextTypes.DEFAULT_TYPE, stop_event: asyncio.Event):
    """Continuously broadcast 'typing...' action until stop_event is signaled."""
    while not stop_event.is_set():
        try:
            await context.bot.send_chat_action(chat_id=chat_id, action=constants.ChatAction.TYPING)
        except Exception:
            pass
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=4.0)
        except asyncio.TimeoutError:
            pass


def execute_pipeline(
    user_query: str,
    resume_text: Optional[str],
    thread_id: str,
    collection_id: Optional[str],
) -> str:
    """Synchronous pipeline runner executed in a background thread."""
    full_response = ""
    for delta in stream_assistant_turn(
        user_query=user_query,
        resume_text=resume_text,
        thread_id=thread_id,
        collection_id=collection_id,
    ):
        full_response += delta
    return full_response


# ============================================================
# Command Handlers
# ============================================================

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start command with an English welcome banner and quick actions."""
    user = update.effective_user
    user_id = str(user.id)
    user_name = user.first_name or "there"

    active_cid = get_user_active_collection(user_id)
    active_coll = get_collection(active_cid) if active_cid else None

    welcome_text = (
        f"👋 **Welcome, {user_name}!**\n\n"
        "I am your **LinkedIn AI Career Assistant & Job Intelligence Bot**.\n\n"
        "Here is what I can do for you:\n"
        "• 🔍 **Autonomous Job Research**: Find 20–40 live, recent jobs across LinkedIn & company portals via Exa.ai.\n"
        "• 🎯 **Tailored Recommendations**: Match your uploaded resume against full job descriptions stored in Qdrant.\n"
        "• 🌐 **Company Intelligence**: Real-time web search for company culture, tech stacks, and active hiring.\n"
        "• ✉️ **Direct Job Application**: Draft personalized cover letters and send applications directly to hiring teams.\n\n"
    )

    if active_coll:
        welcome_text += (
            f"📌 **Active Collection**: `{active_coll.get('name')}`\n"
            f"📊 **Jobs in Collection**: `{active_coll.get('total_jobs', 0)} jobs`\n\n"
            "You can type your query below (e.g., *'Find best RAG jobs above 12L salary'*), "
            "or attach a resume PDF anytime!"
        )
    else:
        welcome_text += (
            "⚠️ **No active collection selected.**\n"
            "To get started, please create your first job search collection using the button below!"
        )

    buttons = [
        [InlineKeyboardButton("➕ Create New Collection", callback_data="cmd_newcollection")],
        [InlineKeyboardButton("📁 My Collections", callback_data="cmd_collections")],
        [InlineKeyboardButton("📎 Attach Resume Info", callback_data="cmd_resume")],
        [InlineKeyboardButton("ℹ️ Help & Commands", callback_data="cmd_help")],
    ]
    reply_markup = InlineKeyboardMarkup(buttons)

    await update.message.reply_text(
        welcome_text,
        parse_mode=constants.ParseMode.MARKDOWN,
        reply_markup=reply_markup,
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Display comprehensive English help menu."""
    help_text = (
        "📖 **LinkedIn AI Assistant — Command Reference**\n\n"
        "**Collection & Job Management**:\n"
        "• `/collections` — List, switch, or delete your job search collections.\n"
        "• `/newcollection` — Create a new collection and fetch 20–40 fresh jobs via Exa.ai.\n"
        "• `/collect_new_jobs` — Fetch additional fresh jobs into your active collection.\n\n"
        "**Resume & Conversation**:\n"
        "• `/resume` — Check your currently attached resume or detach it.\n"
        "• `/newchat` — Start a fresh conversation thread for the active collection.\n"
        "• `/set_gmail <email> <16_digit_password>` — Configure your Gmail for direct email sending.\n\n"
        "**How to Use**:\n"
        "1. Send a **PDF or Image Resume** in chat to automatically attach your profile.\n"
        "2. Type any question, such as:\n"
        "   - *'Find best jobs matching my resume with salary > 1200000'*\n"
        "   - *'Which job is the best fit for me and why?'*\n"
        "   - *'Is Google currently hiring AI Engineers?'*\n"
        "   - *'Apply to Job #1'* (Prepares cover letter with 1-click email send)\n"
    )
    if update.message:
        await update.message.reply_text(help_text, parse_mode=constants.ParseMode.MARKDOWN)
    elif update.callback_query:
        await update.callback_query.message.reply_text(help_text, parse_mode=constants.ParseMode.MARKDOWN)


async def collections_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """List all collections owned by the user with interactive inline actions."""
    user_id = str(update.effective_user.id)
    collections = list_user_collections(user_id)
    active_cid = get_user_active_collection(user_id)

    if not collections:
        msg = (
            "📁 **My Collections**\n\n"
            "You have not created any collections yet. A collection lets you define a specific career role "
            "and automatically gathers 20–40 jobs in a dedicated database collection."
        )
        buttons = [[InlineKeyboardButton("➕ Create New Collection", callback_data="cmd_newcollection")]]
        reply_markup = InlineKeyboardMarkup(buttons)
        if update.message:
            await update.message.reply_text(msg, parse_mode=constants.ParseMode.MARKDOWN, reply_markup=reply_markup)
        else:
            await update.callback_query.message.reply_text(msg, parse_mode=constants.ParseMode.MARKDOWN, reply_markup=reply_markup)
        return

    text = "📁 **Your Job Search Collections**:\n\n"
    keyboard = []

    for idx, c in enumerate(collections, start=1):
        cid = c.get("collection_id") or c.get("project_id")
        cname = c.get("name", "Untitled")
        jobs_count = c.get("total_jobs", 0)
        is_active = (cid == active_cid)
        marker = "🟢 Active" if is_active else "⚪"

        qdrant_name = c.get("qdrant_collection") or c.get("collection_name", "N/A")
        text += f"{idx}. **{cname}** ({marker})\n"
        text += f"   • Jobs: `{jobs_count}` | Vector Store: `{qdrant_name}`\n"
        text += f"   • Query: _{c.get('search_query', 'N/A')[:60]}_\n\n"

        row = []
        if not is_active:
            row.append(InlineKeyboardButton(f"✅ Switch to {cname[:15]}", callback_data=f"switch_coll:{cid}"))
        row.append(InlineKeyboardButton(f"🗑 Delete", callback_data=f"del_coll:{cid}"))
        keyboard.append(row)

    keyboard.append([
        InlineKeyboardButton("➕ Create New Collection", callback_data="cmd_newcollection"),
        InlineKeyboardButton("🔄 Collect Fresh Jobs", callback_data="cmd_collect_jobs"),
    ])

    reply_markup = InlineKeyboardMarkup(keyboard)
    if update.message:
        await update.message.reply_text(text, parse_mode=constants.ParseMode.MARKDOWN, reply_markup=reply_markup)
    else:
        await update.callback_query.message.reply_text(text, parse_mode=constants.ParseMode.MARKDOWN, reply_markup=reply_markup)


# Backward-compatible alias for /projects command
projects_command = collections_command


async def newchat_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Start a fresh conversation thread for the active collection."""
    user_id = str(update.effective_user.id)
    new_thread_id = str(uuid.uuid4())
    set_user_active_thread(user_id, new_thread_id)

    active_cid = get_user_active_collection(user_id)
    active_coll = get_collection(active_cid) if active_cid else None
    coll_name = active_coll.get("name") if active_coll else "Default"

    # Also bind user's resume into this new thread session if one is uploaded
    resume_info = get_user_resume(user_id)
    if resume_info.get("resume_text"):
        session = get_session_state(new_thread_id)
        session["resume_text"] = resume_info["resume_text"]
        session["resume_filename"] = resume_info.get("filename")
        session["resume_file_path"] = resume_info.get("file_path")
        save_session_state(new_thread_id, session, collection_id=active_cid)

    await update.message.reply_text(
        f"🧹 **Started a new conversation!**\n"
        f"Current Collection: `{coll_name}`\n\n"
        "Your previous turn history has been cleared for this new chat. What would you like to search for?",
        parse_mode=constants.ParseMode.MARKDOWN,
    )


async def resume_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show status of attached resume with an option to detach."""
    user_id = str(update.effective_user.id)
    resume_info = get_user_resume(user_id)
    rtext = resume_info.get("resume_text")

    if not rtext:
        text = (
            "📄 **No Resume Attached**\n\n"
            "To attach your resume, simply **send a PDF document or image** into this chat. "
            "I will automatically extract your technical skills, experience, and projects to provide tailored job recommendations."
        )
        if update.message:
            await update.message.reply_text(text, parse_mode=constants.ParseMode.MARKDOWN)
        else:
            await update.callback_query.message.reply_text(text, parse_mode=constants.ParseMode.MARKDOWN)
        return

    fname = resume_info.get("filename", "Uploaded Resume")
    safe_fname = html.escape(fname)
    safe_preview = html.escape(rtext[:400].strip())
    text_html = (
        f"📄 <b>Attached Resume</b>: <code>{safe_fname}</code>\n\n"
        f"<b>Extracted Profile Preview</b>:\n"
        f"<pre>{safe_preview}...</pre>\n\n"
        "Your resume is active and used for all job matches and automatic email cover letters."
    )
    buttons = [[InlineKeyboardButton("❌ Detach Resume", callback_data="detach_resume")]]
    reply_markup = InlineKeyboardMarkup(buttons)

    try:
        if update.message:
            await update.message.reply_text(text_html, parse_mode=constants.ParseMode.HTML, reply_markup=reply_markup)
        else:
            await update.callback_query.message.reply_text(text_html, parse_mode=constants.ParseMode.HTML, reply_markup=reply_markup)
    except Exception:
        plain_text = (
            f"📄 Attached Resume: {fname}\n\n"
            f"Extracted Profile Preview:\n{rtext[:400].strip()}...\n\n"
            "Your resume is active and used for all job matches and automatic email cover letters."
        )
        if update.message:
            await update.message.reply_text(plain_text, reply_markup=reply_markup)
        else:
            await update.callback_query.message.reply_text(plain_text, reply_markup=reply_markup)


async def collect_new_jobs_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Fetch additional fresh jobs via Exa agent into active collection's Qdrant vector store."""
    user_id = str(update.effective_user.id)
    active_cid = get_user_active_collection(user_id)
    active_coll = get_collection(active_cid) if active_cid else None

    if not active_coll:
        msg = "⚠️ Please select or create a collection first using /collections before collecting jobs."
        if update.message:
            await update.message.reply_text(msg)
        else:
            await update.callback_query.message.reply_text(msg)
        return

    status_msg = await (update.message or update.callback_query.message).reply_text(
        f"🔄 **Exa Agent Researching Fresh Jobs** for collection: `{active_coll.get('name')}`...\n"
        "Searching LinkedIn, Naukri, and career pages for 20–40 live postings (this may take ~60–90 seconds).",
        parse_mode=constants.ParseMode.MARKDOWN,
    )

    resume_info = get_user_resume(user_id)
    resume_text = resume_info.get("resume_text")

    def _job_task():
        jobs, metrics = collect_jobs_via_exa(
            query=active_coll.get("search_query") or active_coll.get("name"),
            resume_text=resume_text,
            filters=active_coll.get("search_metadata"),
        )
        coll_name = active_coll.get("qdrant_collection") or active_coll.get("collection_name", "linkedin_jobs")
        store_stats = store_jobs_to_collection(jobs, collection_name=coll_name)
        update_collection(active_cid, {
            "total_jobs": store_stats["total_jobs"],
            "exa_metrics": metrics,
        })
        return store_stats, metrics

    try:
        store_stats, metrics = await asyncio.to_thread(_job_task)
        await status_msg.edit_text(
            f"✅ **Job Collection Complete!**\n\n"
            f"• **New Jobs Added**: `{store_stats['stored_count']}`\n"
            f"• **Duplicates Skipped**: `{store_stats['skipped_count']}`\n"
            f"• **Total Jobs in Collection**: `{store_stats['total_jobs']}`\n"
            f"• **Exa Research Time**: `{metrics.get('duration_seconds', 0)}s`\n\n"
            "You can now ask questions to discover and compare the best roles!",
            parse_mode=constants.ParseMode.MARKDOWN,
        )
    except Exception as e:
        logger.error(f"Error collecting new jobs: {e}", exc_info=True)
        await status_msg.edit_text(f"❌ Failed to collect jobs: {e}")


async def set_gmail_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Allow user to configure their Gmail address and Google App Password for direct dispatch."""
    user_id = str(update.effective_user.id)
    args = context.args

    if not args or len(args) < 2:
        await update.message.reply_text(
            "🔑 **Configure Personal Gmail Dispatch**\n\n"
            "To send job application emails directly from your personal Gmail account, provide your email and 16-digit Google App Password:\n\n"
            "`/set_gmail yourname@gmail.com abcd efgh ijkl mnop`\n\n"
            "ℹ️ *How to get an App Password*:\n"
            "1. Go to Google Account > Security > 2-Step Verification.\n"
            "2. Under App Passwords, create a new one named 'Job Assistant'.",
            parse_mode=constants.ParseMode.MARKDOWN,
        )
        return

    email = args[0]
    app_pass = " ".join(args[1:]).strip()

    if "@" not in email:
        await update.message.reply_text("❌ Invalid email address.")
        return

    set_user_email_creds(user_id, email, app_pass)
    await update.message.reply_text(
        f"✅ **Gmail credentials saved for `{email}`!**\n"
        "You can now send job applications directly from your Gmail account.",
        parse_mode=constants.ParseMode.MARKDOWN,
    )


# ============================================================
# Interactive Wizard: Create New Collection (ConversationHandler)
# ============================================================

async def newcollection_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Step 1: Prompt user for the Collection Name."""
    msg = (
        "🚀 **Create a New Job Search Collection**\n\n"
        "Please enter a **Name** for this collection (e.g., *'AI/ML Roles'*, *'Senior Backend Engineer'*, *'Data Scientist'*):\n\n"
        "_(Send /cancel to abort)_"
    )
    if update.callback_query:
        await update.callback_query.answer()
        await update.callback_query.message.reply_text(msg, parse_mode=constants.ParseMode.MARKDOWN)
    else:
        await update.message.reply_text(msg, parse_mode=constants.ParseMode.MARKDOWN)
    return WAITING_COLLECTION_NAME


async def newcollection_name_received(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Step 2: Save collection name and prompt for search query & criteria."""
    coll_name = update.message.text.strip()
    if not coll_name:
        await update.message.reply_text("Please enter a valid collection name:")
        return WAITING_COLLECTION_NAME

    context.user_data["nc_name"] = coll_name

    await update.message.reply_text(
        f"Great! Collection name set to: **{coll_name}**.\n\n"
        "Now, please describe the **Job Search Criteria & Target Role** in detail:\n"
        "Include: Target role, location preferences, skills, experience level, or salary range.\n\n"
        "*Example*: _'Find AI Engineer and RAG specialist jobs in Bengaluru or Remote, 2-5 years experience, salary above 1500000'_\n\n"
        "_(Send /cancel to abort)_",
        parse_mode=constants.ParseMode.MARKDOWN,
    )
    return WAITING_COLLECTION_CRITERIA


async def newcollection_criteria_received(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Step 3: Execute Exa research agent, create Qdrant collection, save collection, and finish."""
    criteria = update.message.text.strip()
    coll_name = context.user_data.get("nc_name", "New Collection")
    user_id = str(update.effective_user.id)

    status_msg = await update.message.reply_text(
        f"🔍 **Initiating Exa.ai Autonomous Job Discovery** for **'{coll_name}'**...\n"
        "Searching portals for 20–40 live, active positions.\n"
        "⏳ *This process takes about 60–90 seconds. Please wait...*",
        parse_mode=constants.ParseMode.MARKDOWN,
    )

    resume_info = get_user_resume(user_id)
    resume_text = resume_info.get("resume_text")

    def _create_task():
        # 1. Collect jobs via Exa Agent
        jobs, metrics = collect_jobs_via_exa(
            query=criteria,
            resume_text=resume_text,
            filters={"role": coll_name},
        )
        # 2. Create collection entry in Redis
        cid = create_collection(
            name=coll_name,
            search_query=criteria,
            search_metadata={"role": coll_name},
            exa_metrics=metrics,
        )
        coll_obj = get_collection(cid)
        db_coll_name = coll_obj.get("qdrant_collection") or coll_obj.get("collection_name", "linkedin_jobs")

        # 3. Ingest into dedicated Qdrant collection
        store_stats = store_jobs_to_collection(jobs, collection_name=db_coll_name)
        update_collection(cid, {
            "total_jobs": store_stats["total_jobs"],
            "exa_metrics": metrics,
        })
        return cid, store_stats, metrics

    try:
        cid, store_stats, metrics = await asyncio.to_thread(_create_task)

        # Register collection to user and mark as active
        add_user_collection(user_id, cid)
        set_user_active_collection(user_id, cid)

        # Start a new conversation thread for this new collection
        new_tid = str(uuid.uuid4())
        set_user_active_thread(user_id, new_tid)

        await status_msg.edit_text(
            f"🎉 **Collection '{coll_name}' Successfully Created!**\n\n"
            f"• **Active Jobs Collected**: `{store_stats['stored_count']}`\n"
            f"• **Database Collection**: `{get_collection(cid).get('qdrant_collection')}`\n"
            f"• **Exa Research Duration**: `{metrics.get('duration_seconds', 0)}s`\n\n"
            "This collection is now **active**. You can now ask questions to discover the best matches, "
            "compare requirements against your resume, or prepare job applications!",
            parse_mode=constants.ParseMode.MARKDOWN,
        )
    except Exception as e:
        logger.error(f"Failed to create collection: {e}", exc_info=True)
        await status_msg.edit_text(f"❌ Error creating collection: {e}")

    context.user_data.clear()
    return ConversationHandler.END


async def newcollection_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Cancel collection creation wizard."""
    context.user_data.clear()
    await update.message.reply_text("❌ Collection creation cancelled.")
    return ConversationHandler.END


# Backward-compatible aliases for wizard functions
newproject_start = newcollection_start
newproject_name_received = newcollection_name_received
newproject_criteria_received = newcollection_criteria_received
newproject_cancel = newcollection_cancel


# ============================================================
# Document & Resume Upload Handler
# ============================================================

async def document_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle incoming resume files (PDF, image) sent by the user."""
    user = update.effective_user
    user_id = str(user.id)
    doc = update.message.document or (update.message.photo[-1] if update.message.photo else None)

    if not doc:
        return

    os.makedirs("uploads", exist_ok=True)
    filename = getattr(doc, "file_name", None) or f"photo_{doc.file_unique_id}.jpg"
    safe_name = f"tg_{user_id}_{filename.replace(' ', '_')}"
    file_path = os.path.join("uploads", safe_name)

    status_msg = await update.message.reply_text(f"📥 Downloading and parsing `{filename}`...")

    try:
        tg_file = await context.bot.get_file(doc.file_id)
        await tg_file.download_to_drive(custom_path=file_path)

        # Parse text using existing resume_parser
        extracted_text, _ = parse_resume_file(file_path, filename=filename)

        if not extracted_text or len(extracted_text.strip()) < 30:
            await status_msg.edit_text(
                "⚠️ Could not extract readable text from the uploaded document. "
                "Please make sure it is a standard text-based PDF or clear document image."
            )
            return

        # Store in user-scoped resume storage
        resume_data = {
            "resume_text": extracted_text,
            "filename": filename,
            "file_path": os.path.abspath(file_path),
        }
        set_user_resume(user_id, resume_data)

        # Update active thread session state if exists
        active_tid = get_user_active_thread(user_id)
        active_cid = get_user_active_collection(user_id)
        session = get_session_state(active_tid)
        session["resume_text"] = extracted_text
        session["resume_filename"] = filename
        session["resume_file_path"] = os.path.abspath(file_path)
        save_session_state(active_tid, session, collection_id=active_cid)

        preview = extracted_text[:350].strip()
        safe_preview = html.escape(preview)
        safe_filename = html.escape(filename)

        msg_html = (
            f"✅ <b>Resume '<code>{safe_filename}</code>' Attached Successfully!</b>\n\n"
            f"<b>Candidate Summary Preview</b>:\n"
            f"<pre>{safe_preview}...</pre>\n\n"
            "All job matching and personalized application emails will now align strictly with your resume!"
        )
        try:
            await status_msg.edit_text(msg_html, parse_mode=constants.ParseMode.HTML)
        except Exception:
            await status_msg.edit_text(
                f"✅ Resume '{filename}' Attached Successfully!\n\n"
                f"Candidate Summary Preview:\n{preview}...\n\n"
                "All job matching and personalized application emails will now align strictly with your resume!"
            )
    except Exception as e:
        logger.error(f"Failed to process resume document: {e}", exc_info=True)
        await status_msg.edit_text(f"❌ Error parsing resume: {e}")


# ============================================================
# Main Text Message Handler (Pipeline Execution)
# ============================================================

async def text_message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handles user queries, runs the LangGraph pipeline asynchronously,
    and returns tailored recommendations, web intelligence, or application drafts.
    """
    user_query = update.message.text.strip()
    if not user_query:
        return

    user_id = str(update.effective_user.id)
    chat_id = update.effective_chat.id

    # 1. Verify active collection
    active_cid = get_user_active_collection(user_id)
    if not active_cid:
        collections = list_user_collections(user_id)
        if collections:
            buttons = [
                [InlineKeyboardButton(f"Select {c['name']}", callback_data=f"switch_coll:{c.get('collection_id') or c.get('project_id')}")]
                for c in collections[:5]
            ]
            buttons.append([InlineKeyboardButton("➕ Create New Collection", callback_data="cmd_newcollection")])
            await update.message.reply_text(
                "⚠️ **No active collection selected.** Please select an existing collection or create a new one:",
                reply_markup=InlineKeyboardMarkup(buttons),
                parse_mode=constants.ParseMode.MARKDOWN,
            )
        else:
            buttons = [[InlineKeyboardButton("➕ Create New Collection", callback_data="cmd_newcollection")]]
            await update.message.reply_text(
                "⚠️ **No collections found.** Please create your first collection to gather jobs and begin searching:",
                reply_markup=InlineKeyboardMarkup(buttons),
                parse_mode=constants.ParseMode.MARKDOWN,
            )
        return

    active_tid = get_user_active_thread(user_id)
    resume_info = get_user_resume(user_id)
    resume_text = resume_info.get("resume_text")

    # 2. Launch pipeline in worker thread with background typing indicator
    stop_event = asyncio.Event()
    typing_task = asyncio.create_task(send_typing_action(chat_id, context, stop_event))

    try:
        full_response = await asyncio.to_thread(
            execute_pipeline,
            user_query=user_query,
            resume_text=resume_text,
            thread_id=active_tid,
            collection_id=active_cid,
        )
    except Exception as e:
        logger.error(f"Pipeline execution error: {e}", exc_info=True)
        full_response = f"⚠️ An error occurred while processing your request: {e}"
    finally:
        stop_event.set()
        await typing_task

    # 3. Check if turn generated an application email draft
    session = get_session_state(active_tid)
    email_draft = session.get("email_draft")
    has_ready_draft = email_draft and email_draft.get("status") == "ready_to_send"

    # 4. Split and deliver response chunks
    chunks = split_message(full_response)
    for i, chunk in enumerate(chunks):
        # Attach action buttons to the final chunk if an email draft is ready
        if i == len(chunks) - 1 and has_ready_draft:
            comp = email_draft.get("company", "Company")
            buttons = [
                [InlineKeyboardButton(f"✉️ Send via App Mail to {comp}", callback_data="send_email_app")],
                [InlineKeyboardButton(f"🔐 Send via Personal Gmail", callback_data="send_email_gmail")],
            ]
            reply_markup = InlineKeyboardMarkup(buttons)
        else:
            reply_markup = None

        try:
            await update.message.reply_text(
                chunk,
                parse_mode=constants.ParseMode.MARKDOWN,
                reply_markup=reply_markup,
            )
        except Exception:
            # Fallback to plain text if markdown parsing encounters unescaped symbols
            await update.message.reply_text(
                chunk,
                reply_markup=reply_markup,
            )


# ============================================================
# Callback Query Handler (Inline Button Clicks)
# ============================================================

async def callback_query_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Process all inline keyboard actions."""
    query = update.callback_query
    await query.answer()
    data = query.data
    user_id = str(update.effective_user.id)

    if data in ("cmd_newcollection", "cmd_newproject"):
        await query.message.reply_text(
            "🚀 To create a new collection, please type `/newcollection` to launch the step-by-step setup.",
            parse_mode=constants.ParseMode.MARKDOWN,
        )
    elif data in ("cmd_collections", "cmd_projects"):
        await collections_command(update, context)
    elif data == "cmd_resume":
        await resume_command(update, context)
    elif data == "cmd_help":
        await help_command(update, context)
    elif data == "cmd_collect_jobs":
        await collect_new_jobs_command(update, context)
    elif data == "detach_resume":
        clear_user_resume(user_id)
        await query.edit_message_text("✅ Resume detached. You can upload a new resume anytime by sending a PDF file.")
    elif data.startswith("switch_coll:") or data.startswith("switch_proj:"):
        cid = data.split(":", 1)[1]
        coll = get_collection(cid)
        if coll:
            set_user_active_collection(user_id, cid)
            # Create fresh active thread for switched collection
            new_tid = str(uuid.uuid4())
            set_user_active_thread(user_id, new_tid)
            await query.edit_message_text(
                f"✅ Switched active collection to **'{coll.get('name')}'** ({coll.get('total_jobs', 0)} jobs).\n"
                "You can now search and match jobs within this collection!",
                parse_mode=constants.ParseMode.MARKDOWN,
            )
        else:
            await query.edit_message_text("❌ Collection not found.")
    elif data.startswith("del_coll:") or data.startswith("del_proj:"):
        cid = data.split(":", 1)[1]
        coll = get_collection(cid)
        cname = coll.get("name") if coll else "Collection"
        delete_collection(cid)
        remove_user_collection(user_id, cid)
        await query.edit_message_text(
            f"🗑 Collection **'{cname}'** and its Qdrant collection have been deleted.",
            parse_mode=constants.ParseMode.MARKDOWN,
        )
    elif data == "send_email_app":
        # Dispatch application email via App Mail
        active_tid = get_user_active_thread(user_id)
        session = get_session_state(active_tid)
        draft = session.get("email_draft")
        resume_info = get_user_resume(user_id)

        if not draft:
            await query.message.reply_text("❌ No prepared application draft found.")
            return

        recipient = draft.get("recipient_email")
        subject = draft.get("subject", "Job Application")
        body = draft.get("body", "")
        resume_path = resume_info.get("file_path")
        resume_text = resume_info.get("resume_text")
        resume_fname = resume_info.get("filename")

        status_msg = await query.message.reply_text("📤 Sending application email via App Mail...")

        def _send():
            return send_application_email(
                recipient_email=recipient,
                subject=subject,
                body=body,
                sender_option="app",
                user_email=os.getenv("APP_SMTP_USER", "jobassistant1.1@gmail.com"),
                resume_path=resume_path,
                resume_text=resume_text,
                resume_filename=resume_fname,
            )

        res = await asyncio.to_thread(_send)
        if res.get("success"):
            draft["status"] = "sent"
            session["email_draft"] = draft
            save_session_state(active_tid, session)
            await status_msg.edit_text(
                f"🎉 **Application Email Sent Successfully!**\n\n"
                f"• **To**: `{recipient}`\n"
                f"• **Subject**: `{subject}`\n"
                f"• **Resume Attached**: `{resume_fname or 'Included'}`\n\n"
                "Good luck with your application!",
                parse_mode=constants.ParseMode.MARKDOWN,
            )
        else:
            await status_msg.edit_text(f"❌ Failed to send email: {res.get('message')}")

    elif data == "send_email_gmail":
        # Check if user has saved personal Gmail credentials
        creds = get_user_email_creds(user_id)
        if not creds.get("email") or not creds.get("app_password"):
            await query.message.reply_text(
                "🔑 **Personal Gmail Not Configured**\n\n"
                "Please configure your Gmail address and 16-digit Google App Password using:\n"
                "`/set_gmail yourname@gmail.com abcd efgh ijkl mnop`\n\n"
                "Or click **'Send via App Mail'** above to send directly through the system without configuration.",
                parse_mode=constants.ParseMode.MARKDOWN,
            )
            return

        active_tid = get_user_active_thread(user_id)
        session = get_session_state(active_tid)
        draft = session.get("email_draft")
        resume_info = get_user_resume(user_id)

        if not draft:
            await query.message.reply_text("❌ No prepared application draft found.")
            return

        status_msg = await query.message.reply_text(f"📤 Sending application directly from `{creds['email']}`...")

        def _send_gmail():
            return send_application_email(
                recipient_email=draft.get("recipient_email"),
                subject=draft.get("subject", "Job Application"),
                body=draft.get("body", ""),
                sender_option="user_gmail",
                user_email=creds["email"],
                user_app_password=creds["app_password"],
                resume_path=resume_info.get("file_path"),
                resume_text=resume_info.get("resume_text"),
                resume_filename=resume_info.get("filename"),
            )

        res = await asyncio.to_thread(_send_gmail)
        if res.get("success"):
            draft["status"] = "sent"
            session["email_draft"] = draft
            save_session_state(active_tid, session)
            await status_msg.edit_text(
                f"🎉 **Application Sent via Your Gmail ({creds['email']})!**\n\n"
                f"• **Recipient**: `{draft.get('recipient_email')}`\n"
                f"• **Subject**: `{draft.get('subject')}`\n\n"
                "The email has been dispatched from your personal account.",
                parse_mode=constants.ParseMode.MARKDOWN,
            )
        else:
            await status_msg.edit_text(f"❌ Failed to dispatch email via Gmail: {res.get('message')}")


# ============================================================
# Main Application Builder & Runner
# ============================================================

def build_telegram_app() -> Application:
    """Construct and configure the python-telegram-bot Application."""
    if not BOT_TOKEN:
        raise ValueError("BOT_TOKEN is not defined in .env! Please set BOT_TOKEN to run the Telegram Bot.")

    app = Application.builder().token(BOT_TOKEN).build()

    # 1. New Collection Conversation Handler
    newcoll_conv = ConversationHandler(
        entry_points=[
            CommandHandler(["newcollection", "newproject"], newcollection_start),
            CallbackQueryHandler(newcollection_start, pattern="^(cmd_newcollection|cmd_newproject)$"),
        ],
        states={
            WAITING_COLLECTION_NAME: [MessageHandler(filters.TEXT & ~filters.COMMAND, newcollection_name_received)],
            WAITING_COLLECTION_CRITERIA: [MessageHandler(filters.TEXT & ~filters.COMMAND, newcollection_criteria_received)],
        },
        fallbacks=[CommandHandler("cancel", newcollection_cancel)],
        allow_reentry=True,
        per_message=False,
    )
    app.add_handler(newcoll_conv)

    # 2. Command Handlers
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler(["collections", "projects"], collections_command))
    app.add_handler(CommandHandler("newchat", newchat_command))
    app.add_handler(CommandHandler("resume", resume_command))
    app.add_handler(CommandHandler("collect_new_jobs", collect_new_jobs_command))
    app.add_handler(CommandHandler("set_gmail", set_gmail_command))

    # 3. Document / Resume Handler
    app.add_handler(MessageHandler(filters.Document.ALL | filters.PHOTO, document_handler))

    # 4. Callback Query Handler
    app.add_handler(CallbackQueryHandler(callback_query_handler))

    # 5. Fallback Text Message Handler (Pipeline Runner)
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_message_handler))

    return app


if __name__ == "__main__":
    print("=" * 60)
    print("🤖 Launching LinkedIn AI Career Assistant Telegram Bot...")
    print("=" * 60)
    try:
        app = build_telegram_app()
        print("✅ Telegram Bot initialized successfully! Listening for messages...")
        app.run_polling(drop_pending_updates=True)
    except Exception as e:
        print(f"❌ Error starting Telegram bot: {e}")
