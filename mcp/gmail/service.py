"""
Gmail MCP Service for Friday AI
================================

Provides email operations for sending screenplays via Gmail API.
Supports PDF attachments and HTML email body.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import sys
import tempfile
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from mimetypes import guess_type
from pathlib import Path
from typing import List, Optional

LOGGER = logging.getLogger(__name__)

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Gmail API scopes
SCOPES = ["https://www.googleapis.com/auth/gmail.send"]

# Credentials and token paths
CONFIG_DIR = REPO_ROOT / "config"
CREDENTIALS_PATH = CONFIG_DIR / "gmail_credentials.json"
TOKEN_PATH = CONFIG_DIR / "gmail_token.json"


def get_gmail_service():
    """Get authenticated Gmail API service."""
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build
    except ImportError as exc:
        raise RuntimeError(
            "Gmail API dependencies not installed. Run: "
            "pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib"
        ) from exc

    creds = None

    # Load existing token
    if TOKEN_PATH.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), SCOPES)

    # Refresh or create new credentials
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not CREDENTIALS_PATH.exists():
                raise RuntimeError(
                    f"Gmail credentials not found at {CREDENTIALS_PATH}. "
                    "Download from Google Cloud Console and save as gmail_credentials.json"
                )
            flow = InstalledAppFlow.from_client_secrets_file(
                str(CREDENTIALS_PATH), SCOPES
            )
            creds = flow.run_local_server(port=0)

        # Save token for future use
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        with open(TOKEN_PATH, "w") as token:
            token.write(creds.to_json())

    return build("gmail", "v1", credentials=creds)


def send_screenplay_email(
    to: List[str],
    subject: str,
    project_slug: str,
    format: str = "pdf",
    include_html_body: bool = True,
    message_text: Optional[str] = None,
) -> dict:
    """
    Send screenplay via email.

    Args:
        to: List of recipient email addresses
        subject: Email subject line
        project_slug: Screenplay project slug to export
        format: Export format - 'pdf', 'fountain', or 'html'
        include_html_body: Include HTML preview in email body
        message_text: Optional message to include before screenplay

    Returns:
        dict with 'sent': True/False, 'message_id': str
    """
    from scripts.export_screenplay import ScreenplayExporter, get_session

    LOGGER.info(f"Sending screenplay '{project_slug}' to {to}")

    # Get screenplay exporter
    session = get_session()
    try:
        exporter = ScreenplayExporter(project_slug, session)

        # Create message
        message = MIMEMultipart("mixed")
        message["To"] = ", ".join(to)
        message["Subject"] = subject

        # Add text/html body
        if include_html_body:
            html_content = exporter.export_html()
            if message_text:
                html_content = f"<p>{message_text}</p><hr/>" + html_content
            body = MIMEText(html_content, "html", "utf-8")
        else:
            body_text = (
                message_text
                or f"Please find the screenplay '{exporter.project.title}' attached."
            )
            body = MIMEText(body_text, "plain", "utf-8")

        message.attach(body)

        # Add attachment based on format
        if format == "pdf":
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp_path = tmp.name
            try:
                exporter.export_pdf(tmp_path)
                _attach_file(
                    message, tmp_path, f"{project_slug}.pdf", "application/pdf"
                )
            finally:
                os.unlink(tmp_path)

        elif format == "fountain":
            fountain_content = exporter.export_fountain()
            attachment = MIMEText(fountain_content, "plain", "utf-8")
            attachment.add_header(
                "Content-Disposition", "attachment", filename=f"{project_slug}.fountain"
            )
            message.attach(attachment)

        elif format == "html":
            html_content = exporter.export_html()
            attachment = MIMEText(html_content, "html", "utf-8")
            attachment.add_header(
                "Content-Disposition", "attachment", filename=f"{project_slug}.html"
            )
            message.attach(attachment)

        # Send via Gmail API
        service = get_gmail_service()
        raw = base64.urlsafe_b64encode(message.as_bytes()).decode("utf-8")
        result = (
            service.users().messages().send(userId="me", body={"raw": raw}).execute()
        )

        LOGGER.info(f"Email sent successfully. Message ID: {result.get('id')}")
        return {
            "sent": True,
            "message_id": result.get("id"),
            "to": to,
            "subject": subject,
        }

    except Exception as e:
        LOGGER.error(f"Failed to send email: {e}")
        return {
            "sent": False,
            "error": str(e),
        }
    finally:
        session.close()


def _attach_file(message: MIMEMultipart, filepath: str, filename: str, mime_type: str):
    """Attach a file to the email message."""
    with open(filepath, "rb") as f:
        attachment = MIMEBase(*mime_type.split("/"))
        attachment.set_payload(f.read())

    import email.encoders

    email.encoders.encode_base64(attachment)
    attachment.add_header("Content-Disposition", "attachment", filename=filename)
    message.attach(attachment)


def send_simple_email(
    to: List[str],
    subject: str,
    body: str,
    html: bool = False,
) -> dict:
    """
    Send a simple email without attachments.

    Args:
        to: List of recipient email addresses
        subject: Email subject line
        body: Email body content
        html: Whether body is HTML (default: plain text)

    Returns:
        dict with 'sent': True/False, 'message_id': str
    """
    LOGGER.info(f"Sending email to {to}: {subject}")

    try:
        message = MIMEMultipart()
        message["To"] = ", ".join(to)
        message["Subject"] = subject

        content_type = "html" if html else "plain"
        message.attach(MIMEText(body, content_type, "utf-8"))

        service = get_gmail_service()
        raw = base64.urlsafe_b64encode(message.as_bytes()).decode("utf-8")
        result = (
            service.users().messages().send(userId="me", body={"raw": raw}).execute()
        )

        return {
            "sent": True,
            "message_id": result.get("id"),
            "to": to,
            "subject": subject,
        }

    except Exception as e:
        LOGGER.error(f"Failed to send email: {e}")
        return {
            "sent": False,
            "error": str(e),
        }


def list_labels() -> List[dict]:
    """List Gmail labels (for testing API connection)."""
    try:
        service = get_gmail_service()
        results = service.users().labels().list(userId="me").execute()
        labels = results.get("labels", [])
        return [{"id": l["id"], "name": l["name"]} for l in labels]
    except Exception as e:
        LOGGER.error(f"Failed to list labels: {e}")
        return []
