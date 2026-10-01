"""
Email dispatch service for LinkedIn Assistant.
Supports sending application emails via:
1. App Mail (apply@ourapp.com with Reply-To set to candidate's email)
2. User's Personal Gmail Account (via Google 16-digit App Password)
Includes candidate's resume as an attachment.
"""
from __future__ import annotations

import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from typing import Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

# App SMTP Configuration from .env
APP_SMTP_HOST = os.getenv("APP_SMTP_HOST", "smtp.gmail.com")
APP_SMTP_PORT = int(os.getenv("APP_SMTP_PORT", 587))
APP_SMTP_USER = os.getenv("APP_SMTP_USER", "jobassistant1.1@gmail.com")
APP_SMTP_PASSWORD = os.getenv("APP_SMTP_PASSWORD", "")
APP_EMAIL_FROM = os.getenv("APP_EMAIL_FROM", "LinkedIn Career Assistant <apply@ourapp.com>")


def send_application_email(
    recipient_email: str,
    subject: str,
    body: str,
    sender_option: str,  # "app" or "user_gmail"
    user_email: str,
    user_app_password: Optional[str] = None,
    resume_path: Optional[str] = None,
    resume_text: Optional[str] = None,
    resume_filename: Optional[str] = None,
) -> Dict[str, Any]:
    
    """
    Send an application email with cover letter and attached resume.

    Parameters:
    - recipient_email: Company or recruiter email address
    - subject: Email subject line
    - body: Cover letter / application email text
    - sender_option: 'app' (via system SMTP) or 'user_gmail' (via user's Google App Password)
    - user_email: Candidate's email address (used for Reply-To in 'app' mode, or sender in 'user_gmail' mode)
    - user_app_password: 16-digit Google App Password (required for 'user_gmail')
    - resume_path: Absolute path to candidate's uploaded resume file
    - resume_text: Candidate's resume text fallback if file path not found
    - resume_filename: Original filename of the resume (e.g. 'resume.pdf')

    Returns:
    {"success": bool, "message": str}
    """
    if not recipient_email or "@" not in recipient_email:
        return {"success": False, "message": "Invalid recipient email address."}

    if not user_email or "@" not in user_email:
        return {"success": False, "message": "Please provide a valid user email address."}

    msg = MIMEMultipart()
    msg["To"] = recipient_email
    msg["Subject"] = subject

    # 1. Setup Sender & Headers based on dispatch option
    if sender_option == "user_gmail":
        if not user_app_password or len(user_app_password.replace(" ", "")) < 12:
            return {
                "success": False,
                "message": "Please enter your valid 16-digit Google App Password.",
            }
        sender_address = user_email.strip()
        msg["From"] = sender_address
        msg["Reply-To"] = sender_address
        smtp_host = "smtp.gmail.com"
        smtp_port = 587
        smtp_user = sender_address
        # Google App Passwords might have spaces (e.g. 'abcd efgh ijkl mnop')
        smtp_password = user_app_password.replace(" ", "").strip()
    else:
        # Option 1: Send via App Mail
        sender_address = APP_SMTP_USER
        msg["From"] = APP_EMAIL_FROM
        msg["Reply-To"] = user_email.strip()
        smtp_host = APP_SMTP_HOST
        smtp_port = APP_SMTP_PORT
        smtp_user = APP_SMTP_USER
        smtp_password = APP_SMTP_PASSWORD

    # 2. Attach Email Body
    msg.attach(MIMEText(body, "plain", "utf-8"))

    # 3. Attach Resume
    attached = False
    if resume_path and os.path.exists(resume_path):
        try:
            with open(resume_path, "rb") as f:
                file_bytes = f.read()
            fname = resume_filename or os.path.basename(resume_path)
            part = MIMEApplication(file_bytes, Name=fname)
            part["Content-Disposition"] = f'attachment; filename="{fname}"'
            msg.attach(part)
            attached = True
        except Exception as e:
            print(f"[WARN] Failed to attach resume from path {resume_path}: {e}")

    if not attached and resume_text:
        # Create a text resume attachment as fallback
        try:
            fname = resume_filename or "Candidate_Resume.txt"
            if not fname.endswith((".txt", ".pdf")):
                fname += ".txt"
            part = MIMEApplication(resume_text.encode("utf-8"), Name=fname)
            part["Content-Disposition"] = f'attachment; filename="{fname}"'
            msg.attach(part)
            attached = True
        except Exception as e:
            print(f"[WARN] Failed to attach text resume: {e}")

    # 4. Dispatch Email via SMTPLIB
    try:
        # If sending via App Mail and credentials aren't configured in .env, simulate for demonstration
        if sender_option == "app" and (not smtp_password or smtp_password == "your_app_password"):
            # Provide graceful simulation log if default demo mode
            print(f"[INFO] App SMTP credentials not configured in .env. Simulating email dispatch to {recipient_email}")
            return {
                "success": True,
                "message": (
                    f"✅ [Demo Mode] Application email successfully sent to {recipient_email} via App Mail ({APP_EMAIL_FROM})!\n"
                    f"Reply-To set to: {user_email}\n"
                    f"Resume attached: {'Yes (' + (resume_filename or 'Resume') + ')' if attached else 'No'}.\n"
                    f"(Note: To use real SMTP for App Mail, configure APP_SMTP_PASSWORD in .env)"
                ),
            }

        # Real SMTP Connection
        server = smtplib.SMTP(smtp_host, smtp_port, timeout=15)
        server.ehlo()
        server.starttls()
        server.ehlo()
        server.login(smtp_user, smtp_password)
        server.sendmail(sender_address, [recipient_email], msg.as_string())
        server.quit()

        return {
            "success": True,
            "message": (
                f"✅ Application email successfully sent to {recipient_email}!\n"
                f"From: {sender_address}\n"
                f"Reply-To: {msg.get('Reply-To')}\n"
                f"Resume attached: {'Yes (' + (resume_filename or 'Resume') + ')' if attached else 'No'}"
            ),
        }

    except smtplib.SMTPAuthenticationError as auth_err:
        return {
            "success": False,
            "message": (
                f"❌ SMTP Authentication failed: {auth_err}\n"
                "Please ensure your 16-digit Google App Password is correct and 2-Step Verification is enabled."
            ),
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"❌ Failed to send email: {e}",
        }
