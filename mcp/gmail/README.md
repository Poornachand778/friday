# Gmail MCP Tool for Friday AI

Send screenplay exports and emails via Gmail API.

## Setup

### 1. Install Dependencies

```bash
pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib
```

### 2. Get Gmail API Credentials

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing
3. Enable the **Gmail API**
4. Go to **APIs & Services > Credentials**
5. Click **Create Credentials > OAuth client ID**
6. Select **Desktop app**
7. Download the JSON file
8. Save it as `config/gmail_credentials.json`

### 3. First-Time Authorization

Run the test script to authorize:

```bash
python -c "from mcp.gmail import get_gmail_service; get_gmail_service()"
```

This will open a browser for OAuth consent. The token will be saved to `config/gmail_token.json`.

## Usage

### Via MCP Server

Start the server:
```bash
python -m mcp.gmail.server
```

Send JSON-RPC requests:
```json
{"method": "call_tool", "params": {"name": "send_screenplay", "arguments": {"to": ["director@example.com"], "subject": "Script: GUSAGUSALU", "project_slug": "gusagusalu-script", "format": "pdf"}}}
```

### Via Python

```python
from mcp.gmail import send_screenplay_email, send_simple_email

# Send screenplay
result = send_screenplay_email(
    to=["director@example.com", "producer@example.com"],
    subject="Script: GUSAGUSALU - Draft 1",
    project_slug="gusagusalu-script",
    format="pdf",
    include_html_body=True,
    message_text="Please review the attached screenplay."
)

# Send simple email
result = send_simple_email(
    to=["team@example.com"],
    subject="Meeting Notes",
    body="<h1>Notes</h1><p>Discussion points...</p>",
    html=True
)
```

## Tools Available

### send_screenplay

Send screenplay with PDF/Fountain/HTML attachment.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| to | array[string] | Yes | Recipient emails |
| subject | string | Yes | Email subject |
| project_slug | string | Yes | Screenplay project slug |
| format | string | No | pdf, fountain, html (default: pdf) |
| include_html_body | boolean | No | Include HTML preview (default: true) |
| message | string | No | Custom message before screenplay |

### send_email

Send simple email without attachments.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| to | array[string] | Yes | Recipient emails |
| subject | string | Yes | Email subject |
| body | string | Yes | Email body |
| html | boolean | No | Body is HTML (default: false) |

## Security Notes

- `gmail_credentials.json` contains your OAuth client secret - keep it secure
- `gmail_token.json` contains your refresh token - do not share
- Both files are in `.gitignore` to prevent accidental commits
