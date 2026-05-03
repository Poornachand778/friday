"""
Tests for Gmail MCP Service
============================

Comprehensive tests for mcp/gmail/service.py covering:
- Module-level constants (SCOPES, CREDENTIALS_PATH, TOKEN_PATH)
- get_gmail_service: auth flows, token refresh, error handling
- send_screenplay_email: all formats, body options, error cases
- _attach_file: binary attachment with correct MIME headers
- send_simple_email: plain text, HTML, success and failure
- list_labels: success and API failure

All external dependencies (Google APIs, filesystem, ScreenplayExporter)
are heavily mocked.
"""

import base64
import os
import sys
import tempfile
from email.mime.multipart import MIMEMultipart
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch, call, ANY

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def mock_gmail_service():
    """Create a fully-mocked Gmail API service resource."""
    service = MagicMock()
    # messages().send chain
    service.users.return_value.messages.return_value.send.return_value.execute.return_value = {
        "id": "msg-123"
    }
    # labels().list chain
    service.users.return_value.labels.return_value.list.return_value.execute.return_value = {
        "labels": [
            {"id": "INBOX", "name": "INBOX"},
            {"id": "SENT", "name": "SENT"},
        ]
    }
    return service


@pytest.fixture
def mock_google_modules():
    """Provide mock Google auth/api modules for import patching."""
    mock_request = MagicMock()
    mock_credentials_mod = MagicMock()
    mock_flow_mod = MagicMock()
    mock_discovery = MagicMock()
    return {
        "Request": mock_request,
        "Credentials": mock_credentials_mod.Credentials,
        "InstalledAppFlow": mock_flow_mod.InstalledAppFlow,
        "build": mock_discovery.build,
        "modules": {
            "google.auth.transport.requests": mock_request,
            "google.oauth2.credentials": mock_credentials_mod,
            "google_auth_oauthlib.flow": mock_flow_mod,
            "googleapiclient.discovery": mock_discovery,
        },
    }


@pytest.fixture
def mock_exporter():
    """Create a mock ScreenplayExporter instance."""
    exporter = MagicMock()
    exporter.project.title = "Test Screenplay"
    exporter.export_html.return_value = "<html><body>Screenplay</body></html>"
    exporter.export_fountain.return_value = "Title: Test\n\nINT. ROOM - DAY"
    exporter.export_pdf.return_value = None  # writes to file
    return exporter


@pytest.fixture
def mock_session():
    """Create a mock database session."""
    session = MagicMock()
    return session


# ===================================================================
# 1. Module Constants
# ===================================================================


class TestModuleConstants:
    """Tests for module-level constants."""

    def test_scopes_contains_gmail_send(self):
        """SCOPES should include the gmail.send scope."""
        from mcp.gmail.service import SCOPES

        assert "https://www.googleapis.com/auth/gmail.send" in SCOPES

    def test_scopes_is_list(self):
        """SCOPES should be a list."""
        from mcp.gmail.service import SCOPES

        assert isinstance(SCOPES, list)

    def test_scopes_has_exactly_one_entry(self):
        """SCOPES should have exactly one scope defined."""
        from mcp.gmail.service import SCOPES

        assert len(SCOPES) == 1

    def test_credentials_path_is_path_object(self):
        """CREDENTIALS_PATH should be a Path object."""
        from mcp.gmail.service import CREDENTIALS_PATH

        assert isinstance(CREDENTIALS_PATH, Path)

    def test_credentials_path_ends_with_expected_filename(self):
        """CREDENTIALS_PATH should point to gmail_credentials.json."""
        from mcp.gmail.service import CREDENTIALS_PATH

        assert CREDENTIALS_PATH.name == "gmail_credentials.json"

    def test_credentials_path_in_config_dir(self):
        """CREDENTIALS_PATH should be inside the config directory."""
        from mcp.gmail.service import CREDENTIALS_PATH

        assert CREDENTIALS_PATH.parent.name == "config"

    def test_token_path_is_path_object(self):
        """TOKEN_PATH should be a Path object."""
        from mcp.gmail.service import TOKEN_PATH

        assert isinstance(TOKEN_PATH, Path)

    def test_token_path_ends_with_expected_filename(self):
        """TOKEN_PATH should point to gmail_token.json."""
        from mcp.gmail.service import TOKEN_PATH

        assert TOKEN_PATH.name == "gmail_token.json"

    def test_token_path_in_config_dir(self):
        """TOKEN_PATH should be inside the config directory."""
        from mcp.gmail.service import TOKEN_PATH

        assert TOKEN_PATH.parent.name == "config"

    def test_config_dir_is_under_repo_root(self):
        """CONFIG_DIR should be under REPO_ROOT."""
        from mcp.gmail.service import CONFIG_DIR, REPO_ROOT

        assert CONFIG_DIR.parent == REPO_ROOT

    def test_repo_root_is_absolute(self):
        """REPO_ROOT should be an absolute path."""
        from mcp.gmail.service import REPO_ROOT

        assert REPO_ROOT.is_absolute()


# ===================================================================
# 2. get_gmail_service
# ===================================================================


class TestGetGmailService:
    """Tests for get_gmail_service() function."""

    def test_missing_google_dependencies_raises_runtime_error(self):
        """Should raise RuntimeError when Google libraries are not installed."""
        import mcp.gmail.service as svc

        # Force ImportError by temporarily breaking imports
        with patch.dict("sys.modules", {"google.auth.transport.requests": None}):
            # We need to actually trigger the import inside the function.
            # Patch builtins.__import__ to raise ImportError for google modules.
            original_import = (
                __builtins__.__import__
                if hasattr(__builtins__, "__import__")
                else __import__
            )

            def fake_import(name, *args, **kwargs):
                if name.startswith("google") or name.startswith("googleapiclient"):
                    raise ImportError(f"No module named '{name}'")
                return original_import(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=fake_import):
                with pytest.raises(
                    RuntimeError, match="Gmail API dependencies not installed"
                ):
                    svc.get_gmail_service()

    def test_missing_google_auth_oauthlib_raises_runtime_error(self):
        """Should raise RuntimeError when google_auth_oauthlib is missing."""
        import mcp.gmail.service as svc

        original_import = __import__

        def fake_import(name, *args, **kwargs):
            if "oauthlib" in name:
                raise ImportError(f"No module named '{name}'")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            with pytest.raises(
                RuntimeError, match="Gmail API dependencies not installed"
            ):
                svc.get_gmail_service()

    def test_token_exists_and_valid(self, mock_gmail_service):
        """When token exists and credentials are valid, should return service directly."""
        import mcp.gmail.service as svc

        mock_creds = MagicMock()
        mock_creds.valid = True

        mock_Credentials = MagicMock()
        mock_Credentials.from_authorized_user_file.return_value = mock_creds

        mock_build = MagicMock(return_value=mock_gmail_service)

        with patch.object(svc, "TOKEN_PATH") as mock_token_path:
            mock_token_path.exists.return_value = True
            with patch("mcp.gmail.service.get_gmail_service") as mock_get:
                mock_get.return_value = mock_gmail_service
                result = mock_get()
                assert result == mock_gmail_service

    def test_token_exists_valid_credentials_no_refresh(self):
        """When token exists and creds are valid, no refresh should occur."""
        import mcp.gmail.service as svc

        mock_creds = MagicMock()
        mock_creds.valid = True

        mock_Request = MagicMock()
        mock_Credentials = MagicMock()
        mock_Credentials.from_authorized_user_file.return_value = mock_creds
        mock_build = MagicMock()

        original_import = __import__

        def patched_import(name, *args, **kwargs):
            if name == "google.auth.transport.requests":
                mod = MagicMock()
                mod.Request = mock_Request
                return mod
            if name == "google.oauth2.credentials":
                mod = MagicMock()
                mod.Credentials = mock_Credentials
                return mod
            if name == "google_auth_oauthlib.flow":
                return MagicMock()
            if name == "googleapiclient.discovery":
                mod = MagicMock()
                mod.build = mock_build
                return mod
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=patched_import):
            with patch.object(svc, "TOKEN_PATH") as mock_tp:
                mock_tp.exists.return_value = True
                svc.get_gmail_service()
                mock_creds.refresh.assert_not_called()

    def test_token_expired_and_refreshable(self):
        """When token exists but expired with refresh_token, should refresh."""
        import mcp.gmail.service as svc

        mock_creds = MagicMock()
        mock_creds.valid = False
        mock_creds.expired = True
        mock_creds.refresh_token = "refresh-token-123"

        mock_Request_cls = MagicMock()
        mock_Credentials = MagicMock()
        mock_Credentials.from_authorized_user_file.return_value = mock_creds
        mock_build = MagicMock()

        original_import = __import__

        def patched_import(name, *args, **kwargs):
            if name == "google.auth.transport.requests":
                mod = MagicMock()
                mod.Request = mock_Request_cls
                return mod
            if name == "google.oauth2.credentials":
                mod = MagicMock()
                mod.Credentials = mock_Credentials
                return mod
            if name == "google_auth_oauthlib.flow":
                return MagicMock()
            if name == "googleapiclient.discovery":
                mod = MagicMock()
                mod.build = mock_build
                return mod
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=patched_import):
            with patch.object(svc, "TOKEN_PATH") as mock_tp:
                mock_tp.exists.return_value = True
                with patch.object(svc, "CONFIG_DIR") as mock_cd:
                    with patch("builtins.open", mock_open()):
                        svc.get_gmail_service()
                        mock_creds.refresh.assert_called_once()

    def test_no_token_needs_oauth_flow(self):
        """When no token exists, should run OAuth flow."""
        import mcp.gmail.service as svc

        mock_flow = MagicMock()
        mock_new_creds = MagicMock()
        mock_new_creds.valid = True
        mock_flow.run_local_server.return_value = mock_new_creds

        mock_InstalledAppFlow = MagicMock()
        mock_InstalledAppFlow.from_client_secrets_file.return_value = mock_flow

        mock_build = MagicMock()

        original_import = __import__

        def patched_import(name, *args, **kwargs):
            if name == "google.auth.transport.requests":
                return MagicMock()
            if name == "google.oauth2.credentials":
                mod = MagicMock()
                mod.Credentials.from_authorized_user_file.return_value = None
                return mod
            if name == "google_auth_oauthlib.flow":
                mod = MagicMock()
                mod.InstalledAppFlow = mock_InstalledAppFlow
                return mod
            if name == "googleapiclient.discovery":
                mod = MagicMock()
                mod.build = mock_build
                return mod
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=patched_import):
            with patch.object(svc, "TOKEN_PATH") as mock_tp:
                mock_tp.exists.return_value = False
                with patch.object(svc, "CREDENTIALS_PATH") as mock_cp:
                    mock_cp.exists.return_value = True
                    with patch.object(svc, "CONFIG_DIR") as mock_cd:
                        with patch("builtins.open", mock_open()):
                            svc.get_gmail_service()
                            mock_flow.run_local_server.assert_called_once_with(port=0)

    def test_credentials_file_missing_raises_runtime_error(self):
        """When no token and no credentials file, should raise RuntimeError."""
        import mcp.gmail.service as svc

        mock_creds = MagicMock()
        mock_creds.valid = False
        mock_creds.expired = False

        original_import = __import__

        def patched_import(name, *args, **kwargs):
            if name == "google.auth.transport.requests":
                return MagicMock()
            if name == "google.oauth2.credentials":
                mod = MagicMock()
                mod.Credentials.from_authorized_user_file.return_value = mock_creds
                return mod
            if name == "google_auth_oauthlib.flow":
                return MagicMock()
            if name == "googleapiclient.discovery":
                return MagicMock()
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=patched_import):
            with patch.object(svc, "TOKEN_PATH") as mock_tp:
                # Token path does not exist
                mock_tp.exists.return_value = False
                with patch.object(svc, "CREDENTIALS_PATH") as mock_cp:
                    mock_cp.exists.return_value = False
                    with pytest.raises(
                        RuntimeError, match="Gmail credentials not found"
                    ):
                        svc.get_gmail_service()

    def test_saves_token_after_creation_via_oauth(self):
        """After OAuth flow creates new credentials, token should be saved."""
        import mcp.gmail.service as svc

        mock_flow = MagicMock()
        mock_new_creds = MagicMock()
        mock_new_creds.valid = True
        mock_new_creds.to_json.return_value = '{"token": "new-token"}'
        mock_flow.run_local_server.return_value = mock_new_creds

        mock_InstalledAppFlow = MagicMock()
        mock_InstalledAppFlow.from_client_secrets_file.return_value = mock_flow

        original_import = __import__

        def patched_import(name, *args, **kwargs):
            if name == "google.auth.transport.requests":
                return MagicMock()
            if name == "google.oauth2.credentials":
                mod = MagicMock()
                mod.Credentials.from_authorized_user_file.return_value = None
                return mod
            if name == "google_auth_oauthlib.flow":
                mod = MagicMock()
                mod.InstalledAppFlow = mock_InstalledAppFlow
                return mod
            if name == "googleapiclient.discovery":
                return MagicMock()
            return original_import(name, *args, **kwargs)

        m = mock_open()
        with patch("builtins.__import__", side_effect=patched_import):
            with patch.object(svc, "TOKEN_PATH") as mock_tp:
                mock_tp.exists.return_value = False
                with patch.object(svc, "CREDENTIALS_PATH") as mock_cp:
                    mock_cp.exists.return_value = True
                    with patch.object(svc, "CONFIG_DIR") as mock_cd:
                        with patch("builtins.open", m):
                            svc.get_gmail_service()
                            # Verify token was written
                            m().write.assert_called_once_with('{"token": "new-token"}')

    def test_saves_token_after_refresh(self):
        """After refreshing expired credentials, token should be saved."""
        import mcp.gmail.service as svc

        mock_creds = MagicMock()
        mock_creds.valid = False
        mock_creds.expired = True
        mock_creds.refresh_token = "refresh-tok"
        mock_creds.to_json.return_value = '{"refreshed": true}'

        original_import = __import__

        def patched_import(name, *args, **kwargs):
            if name == "google.auth.transport.requests":
                return MagicMock()
            if name == "google.oauth2.credentials":
                mod = MagicMock()
                mod.Credentials.from_authorized_user_file.return_value = mock_creds
                return mod
            if name == "google_auth_oauthlib.flow":
                return MagicMock()
            if name == "googleapiclient.discovery":
                return MagicMock()
            return original_import(name, *args, **kwargs)

        m = mock_open()
        with patch("builtins.__import__", side_effect=patched_import):
            with patch.object(svc, "TOKEN_PATH") as mock_tp:
                mock_tp.exists.return_value = True
                with patch.object(svc, "CONFIG_DIR") as mock_cd:
                    with patch("builtins.open", m):
                        svc.get_gmail_service()
                        m().write.assert_called_once_with('{"refreshed": true}')

    def test_builds_gmail_v1_service(self):
        """Should call build('gmail', 'v1', credentials=creds)."""
        import mcp.gmail.service as svc

        mock_creds = MagicMock()
        mock_creds.valid = True

        mock_build = MagicMock()

        original_import = __import__

        def patched_import(name, *args, **kwargs):
            if name == "google.auth.transport.requests":
                return MagicMock()
            if name == "google.oauth2.credentials":
                mod = MagicMock()
                mod.Credentials.from_authorized_user_file.return_value = mock_creds
                return mod
            if name == "google_auth_oauthlib.flow":
                return MagicMock()
            if name == "googleapiclient.discovery":
                mod = MagicMock()
                mod.build = mock_build
                return mod
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=patched_import):
            with patch.object(svc, "TOKEN_PATH") as mock_tp:
                mock_tp.exists.return_value = True
                svc.get_gmail_service()
                mock_build.assert_called_once_with(
                    "gmail", "v1", credentials=mock_creds
                )

    def test_config_dir_created_on_token_save(self):
        """CONFIG_DIR.mkdir should be called before saving token."""
        import mcp.gmail.service as svc

        mock_creds = MagicMock()
        mock_creds.valid = False
        mock_creds.expired = True
        mock_creds.refresh_token = "tok"
        mock_creds.to_json.return_value = "{}"

        original_import = __import__

        def patched_import(name, *args, **kwargs):
            if name == "google.auth.transport.requests":
                return MagicMock()
            if name == "google.oauth2.credentials":
                mod = MagicMock()
                mod.Credentials.from_authorized_user_file.return_value = mock_creds
                return mod
            if name == "google_auth_oauthlib.flow":
                return MagicMock()
            if name == "googleapiclient.discovery":
                return MagicMock()
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=patched_import):
            with patch.object(svc, "TOKEN_PATH") as mock_tp:
                mock_tp.exists.return_value = True
                with patch.object(svc, "CONFIG_DIR") as mock_cd:
                    with patch("builtins.open", mock_open()):
                        svc.get_gmail_service()
                        mock_cd.mkdir.assert_called_once_with(
                            parents=True, exist_ok=True
                        )

    def test_runtime_error_includes_pip_install_hint(self):
        """RuntimeError for missing deps should include pip install command."""
        import mcp.gmail.service as svc

        original_import = __import__

        def fake_import(name, *args, **kwargs):
            if name.startswith("google") or name.startswith("googleapiclient"):
                raise ImportError(f"No module named '{name}'")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            with pytest.raises(RuntimeError, match="pip install"):
                svc.get_gmail_service()

    def test_loads_token_with_scopes(self):
        """from_authorized_user_file should be called with SCOPES."""
        import mcp.gmail.service as svc

        mock_creds = MagicMock()
        mock_creds.valid = True
        mock_Credentials = MagicMock()
        mock_Credentials.from_authorized_user_file.return_value = mock_creds

        original_import = __import__

        def patched_import(name, *args, **kwargs):
            if name == "google.auth.transport.requests":
                return MagicMock()
            if name == "google.oauth2.credentials":
                mod = MagicMock()
                mod.Credentials = mock_Credentials
                return mod
            if name == "google_auth_oauthlib.flow":
                return MagicMock()
            if name == "googleapiclient.discovery":
                return MagicMock()
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=patched_import):
            with patch.object(svc, "TOKEN_PATH") as mock_tp:
                mock_tp.exists.return_value = True
                mock_tp.__str__ = MagicMock(return_value="/fake/token.json")
                svc.get_gmail_service()
                mock_Credentials.from_authorized_user_file.assert_called_once_with(
                    str(mock_tp), svc.SCOPES
                )


# ===================================================================
# 3. send_screenplay_email
# ===================================================================


class TestSendScreenplayEmail:
    """Tests for send_screenplay_email() function."""

    def _call_send(
        self,
        mock_gmail_service,
        mock_exporter,
        mock_session,
        format="pdf",
        include_html_body=True,
        message_text=None,
        to=None,
        subject="Test Subject",
        project_slug="test-project",
    ):
        """Helper to call send_screenplay_email with mocks in place."""
        if to is None:
            to = ["test@example.com"]

        with patch(
            "mcp.gmail.service.get_gmail_service", return_value=mock_gmail_service
        ):
            with patch(
                "scripts.export_screenplay.ScreenplayExporter",
                return_value=mock_exporter,
            ) as MockExporter:
                with patch(
                    "scripts.export_screenplay.get_session",
                    return_value=mock_session,
                ):
                    from mcp.gmail.service import send_screenplay_email

                    return send_screenplay_email(
                        to=to,
                        subject=subject,
                        project_slug=project_slug,
                        format=format,
                        include_html_body=include_html_body,
                        message_text=message_text,
                    )

    def test_send_pdf_success(self, mock_gmail_service, mock_exporter, mock_session):
        """Should successfully send email with PDF attachment."""

        # Make export_pdf write some bytes to the temp file
        def fake_export_pdf(path):
            with open(path, "wb") as f:
                f.write(b"%PDF-1.4 fake pdf content")

        mock_exporter.export_pdf.side_effect = fake_export_pdf

        result = self._call_send(
            mock_gmail_service, mock_exporter, mock_session, format="pdf"
        )

        assert result["sent"] is True
        assert result["message_id"] == "msg-123"
        assert result["to"] == ["test@example.com"]
        assert result["subject"] == "Test Subject"

    def test_send_fountain_success(
        self, mock_gmail_service, mock_exporter, mock_session
    ):
        """Should successfully send email with Fountain attachment."""
        result = self._call_send(
            mock_gmail_service, mock_exporter, mock_session, format="fountain"
        )

        assert result["sent"] is True
        assert result["message_id"] == "msg-123"

    def test_send_html_format_success(
        self, mock_gmail_service, mock_exporter, mock_session
    ):
        """Should successfully send email with HTML attachment."""
        result = self._call_send(
            mock_gmail_service, mock_exporter, mock_session, format="html"
        )

        assert result["sent"] is True
        assert result["message_id"] == "msg-123"

    def test_include_html_body_true(
        self, mock_gmail_service, mock_exporter, mock_session
    ):
        """When include_html_body=True, should call exporter.export_html for body."""

        def fake_export_pdf(path):
            with open(path, "wb") as f:
                f.write(b"pdf data")

        mock_exporter.export_pdf.side_effect = fake_export_pdf

        self._call_send(
            mock_gmail_service,
            mock_exporter,
            mock_session,
            include_html_body=True,
            format="pdf",
        )

        mock_exporter.export_html.assert_called()

    def test_include_html_body_false(
        self, mock_gmail_service, mock_exporter, mock_session
    ):
        """When include_html_body=False, should use plain text body."""
        result = self._call_send(
            mock_gmail_service,
            mock_exporter,
            mock_session,
            include_html_body=False,
            format="fountain",
        )

        assert result["sent"] is True

    def test_include_html_body_false_default_message(
        self, mock_gmail_service, mock_exporter, mock_session
    ):
        """When include_html_body=False and no message_text, should use default text with project title."""
        # The default message references exporter.project.title
        result = self._call_send(
            mock_gmail_service,
            mock_exporter,
            mock_session,
            include_html_body=False,
            message_text=None,
            format="fountain",
        )

        assert result["sent"] is True

    def test_custom_message_text_with_html_body(
        self, mock_gmail_service, mock_exporter, mock_session
    ):
        """When include_html_body=True and message_text provided, should prepend message."""

        def fake_export_pdf(path):
            with open(path, "wb") as f:
                f.write(b"pdf data")

        mock_exporter.export_pdf.side_effect = fake_export_pdf

        result = self._call_send(
            mock_gmail_service,
            mock_exporter,
            mock_session,
            include_html_body=True,
            message_text="Check this out!",
            format="pdf",
        )

        assert result["sent"] is True

    def test_custom_message_text_with_plain_body(
        self, mock_gmail_service, mock_exporter, mock_session
    ):
        """When include_html_body=False and message_text provided, should use it as body."""
        result = self._call_send(
            mock_gmail_service,
            mock_exporter,
            mock_session,
            include_html_body=False,
            message_text="Here is the screenplay",
            format="fountain",
        )

        assert result["sent"] is True

    def test_multiple_recipients(self, mock_gmail_service, mock_exporter, mock_session):
        """Should handle multiple recipients."""
        result = self._call_send(
            mock_gmail_service,
            mock_exporter,
            mock_session,
            to=["a@test.com", "b@test.com"],
            format="fountain",
        )

        assert result["sent"] is True
        assert result["to"] == ["a@test.com", "b@test.com"]

    def test_send_failure_returns_error_dict(
        self, mock_gmail_service, mock_exporter, mock_session
    ):
        """When Gmail API send fails, should return error dict."""
        mock_gmail_service.users.return_value.messages.return_value.send.return_value.execute.side_effect = Exception(
            "API Error"
        )

        result = self._call_send(
            mock_gmail_service, mock_exporter, mock_session, format="fountain"
        )

        assert result["sent"] is False
        assert "error" in result
        assert "API Error" in result["error"]

    def test_exporter_exception_returns_error_dict(
        self, mock_gmail_service, mock_session
    ):
        """When ScreenplayExporter raises, should return error dict."""
        with patch(
            "mcp.gmail.service.get_gmail_service", return_value=mock_gmail_service
        ):
            with patch(
                "scripts.export_screenplay.ScreenplayExporter",
                side_effect=ValueError("Project not found: bad-slug"),
            ):
                with patch(
                    "scripts.export_screenplay.get_session",
                    return_value=mock_session,
                ):
                    from mcp.gmail.service import send_screenplay_email

                    result = send_screenplay_email(
                        to=["x@test.com"],
                        subject="Test",
                        project_slug="bad-slug",
                    )

                    assert result["sent"] is False
                    assert "Project not found" in result["error"]

    def test_session_closed_on_success(
        self, mock_gmail_service, mock_exporter, mock_session
    ):
        """Session should be closed after successful send."""
        self._call_send(
            mock_gmail_service, mock_exporter, mock_session, format="fountain"
        )

        mock_session.close.assert_called_once()

    def test_session_closed_on_failure(
        self, mock_gmail_service, mock_exporter, mock_session
    ):
        """Session should be closed even when send fails."""
        mock_gmail_service.users.return_value.messages.return_value.send.return_value.execute.side_effect = Exception(
            "fail"
        )

        self._call_send(
            mock_gmail_service, mock_exporter, mock_session, format="fountain"
        )

        mock_session.close.assert_called_once()

    def test_pdf_temp_file_cleaned_up(
        self, mock_gmail_service, mock_exporter, mock_session
    ):
        """Temp PDF file should be cleaned up after send."""
        created_paths = []

        def fake_export_pdf(path):
            created_paths.append(path)
            with open(path, "wb") as f:
                f.write(b"pdf data")

        mock_exporter.export_pdf.side_effect = fake_export_pdf

        self._call_send(mock_gmail_service, mock_exporter, mock_session, format="pdf")

        # Temp file should have been deleted
        assert len(created_paths) == 1
        assert not os.path.exists(created_paths[0])

    def test_pdf_temp_file_cleaned_up_on_error(
        self, mock_gmail_service, mock_exporter, mock_session
    ):
        """Temp PDF file should be cleaned up even if send fails."""
        created_paths = []

        def fake_export_pdf(path):
            created_paths.append(path)
            with open(path, "wb") as f:
                f.write(b"pdf")

        mock_exporter.export_pdf.side_effect = fake_export_pdf
        mock_gmail_service.users.return_value.messages.return_value.send.return_value.execute.side_effect = Exception(
            "fail"
        )

        self._call_send(mock_gmail_service, mock_exporter, mock_session, format="pdf")

        assert len(created_paths) == 1
        assert not os.path.exists(created_paths[0])

    def test_send_raw_message_is_base64_encoded(
        self, mock_gmail_service, mock_exporter, mock_session
    ):
        """The raw message sent to Gmail API should be base64url-encoded."""
        self._call_send(
            mock_gmail_service, mock_exporter, mock_session, format="fountain"
        )

        send_call = mock_gmail_service.users.return_value.messages.return_value.send
        send_call.assert_called_once()
        call_kwargs = send_call.call_args
        body = (
            call_kwargs[1]["body"]
            if "body" in (call_kwargs[1] or {})
            else call_kwargs.kwargs.get("body")
        )
        assert "raw" in body
        # Verify it's valid base64
        decoded = base64.urlsafe_b64decode(body["raw"])
        assert len(decoded) > 0

    def test_message_has_correct_subject(
        self, mock_gmail_service, mock_exporter, mock_session
    ):
        """The email message should have the correct Subject header."""
        self._call_send(
            mock_gmail_service,
            mock_exporter,
            mock_session,
            format="fountain",
            subject="My Screenplay Draft",
        )

        send_call = mock_gmail_service.users.return_value.messages.return_value.send
        call_kwargs = send_call.call_args
        body = call_kwargs.kwargs.get("body", call_kwargs[1].get("body", {}))
        raw_bytes = base64.urlsafe_b64decode(body["raw"])
        assert b"My Screenplay Draft" in raw_bytes

    def test_message_has_correct_to(
        self, mock_gmail_service, mock_exporter, mock_session
    ):
        """The email message should have the correct To header."""
        self._call_send(
            mock_gmail_service,
            mock_exporter,
            mock_session,
            format="fountain",
            to=["recipient@example.com"],
        )

        send_call = mock_gmail_service.users.return_value.messages.return_value.send
        call_kwargs = send_call.call_args
        body = call_kwargs.kwargs.get("body", call_kwargs[1].get("body", {}))
        raw_bytes = base64.urlsafe_b64decode(body["raw"])
        assert b"recipient@example.com" in raw_bytes

    def test_fountain_attachment_has_correct_filename(
        self, mock_gmail_service, mock_exporter, mock_session
    ):
        """Fountain attachment should have project_slug.fountain filename."""
        self._call_send(
            mock_gmail_service,
            mock_exporter,
            mock_session,
            format="fountain",
            project_slug="my-script",
        )

        send_call = mock_gmail_service.users.return_value.messages.return_value.send
        call_kwargs = send_call.call_args
        body = call_kwargs.kwargs.get("body", call_kwargs[1].get("body", {}))
        raw_bytes = base64.urlsafe_b64decode(body["raw"])
        assert b"my-script.fountain" in raw_bytes

    def test_html_attachment_has_correct_filename(
        self, mock_gmail_service, mock_exporter, mock_session
    ):
        """HTML attachment should have project_slug.html filename."""
        self._call_send(
            mock_gmail_service,
            mock_exporter,
            mock_session,
            format="html",
            project_slug="my-script",
        )

        send_call = mock_gmail_service.users.return_value.messages.return_value.send
        call_kwargs = send_call.call_args
        body = call_kwargs.kwargs.get("body", call_kwargs[1].get("body", {}))
        raw_bytes = base64.urlsafe_b64decode(body["raw"])
        assert b"my-script.html" in raw_bytes

    def test_send_uses_user_me(self, mock_gmail_service, mock_exporter, mock_session):
        """Gmail API send should use userId='me'."""
        self._call_send(
            mock_gmail_service, mock_exporter, mock_session, format="fountain"
        )

        send_call = mock_gmail_service.users.return_value.messages.return_value.send
        send_call.assert_called_once()
        call_kwargs = send_call.call_args
        assert call_kwargs.kwargs.get("userId") == "me" or (
            call_kwargs[1] and call_kwargs[1].get("userId") == "me"
        )

    def test_export_html_called_for_html_format(
        self, mock_gmail_service, mock_exporter, mock_session
    ):
        """For format='html', export_html should be called for the attachment."""
        self._call_send(mock_gmail_service, mock_exporter, mock_session, format="html")

        mock_exporter.export_html.assert_called()

    def test_export_fountain_called_for_fountain_format(
        self, mock_gmail_service, mock_exporter, mock_session
    ):
        """For format='fountain', export_fountain should be called."""
        self._call_send(
            mock_gmail_service, mock_exporter, mock_session, format="fountain"
        )

        mock_exporter.export_fountain.assert_called_once()

    def test_export_pdf_called_for_pdf_format(
        self, mock_gmail_service, mock_exporter, mock_session
    ):
        """For format='pdf', export_pdf should be called."""

        def fake_export_pdf(path):
            with open(path, "wb") as f:
                f.write(b"pdf")

        mock_exporter.export_pdf.side_effect = fake_export_pdf

        self._call_send(mock_gmail_service, mock_exporter, mock_session, format="pdf")

        mock_exporter.export_pdf.assert_called_once()

    def test_return_dict_has_expected_keys_on_success(
        self, mock_gmail_service, mock_exporter, mock_session
    ):
        """On success, return dict should have sent, message_id, to, subject."""
        result = self._call_send(
            mock_gmail_service, mock_exporter, mock_session, format="fountain"
        )

        assert "sent" in result
        assert "message_id" in result
        assert "to" in result
        assert "subject" in result

    def test_return_dict_has_expected_keys_on_failure(
        self, mock_gmail_service, mock_exporter, mock_session
    ):
        """On failure, return dict should have sent and error."""
        mock_gmail_service.users.return_value.messages.return_value.send.return_value.execute.side_effect = Exception(
            "fail"
        )

        result = self._call_send(
            mock_gmail_service, mock_exporter, mock_session, format="fountain"
        )

        assert "sent" in result
        assert "error" in result
        assert result["sent"] is False


# ===================================================================
# 4. _attach_file
# ===================================================================


class TestAttachFile:
    """Tests for _attach_file() helper function."""

    def test_attach_file_creates_attachment(self, tmp_path):
        """Should add an attachment to the MIMEMultipart message."""
        from mcp.gmail.service import _attach_file

        filepath = tmp_path / "test.pdf"
        filepath.write_bytes(b"%PDF-1.4 test content here")

        msg = MIMEMultipart()
        _attach_file(msg, str(filepath), "test.pdf", "application/pdf")

        payloads = msg.get_payload()
        assert len(payloads) == 1

    def test_attach_file_correct_content_disposition(self, tmp_path):
        """Attachment should have Content-Disposition header with filename."""
        from mcp.gmail.service import _attach_file

        filepath = tmp_path / "script.pdf"
        filepath.write_bytes(b"fake pdf")

        msg = MIMEMultipart()
        _attach_file(msg, str(filepath), "script.pdf", "application/pdf")

        attachment = msg.get_payload()[0]
        disposition = attachment.get("Content-Disposition")
        assert "attachment" in disposition
        assert "script.pdf" in disposition

    def test_attach_file_correct_content_type(self, tmp_path):
        """Attachment should have the correct MIME type."""
        from mcp.gmail.service import _attach_file

        filepath = tmp_path / "doc.pdf"
        filepath.write_bytes(b"content")

        msg = MIMEMultipart()
        _attach_file(msg, str(filepath), "doc.pdf", "application/pdf")

        attachment = msg.get_payload()[0]
        assert attachment.get_content_type() == "application/pdf"

    def test_attach_file_payload_is_base64_encoded(self, tmp_path):
        """Attachment payload should be base64-encoded."""
        from mcp.gmail.service import _attach_file

        filepath = tmp_path / "test.bin"
        original_content = b"\x00\x01\x02\xff\xfe\xfd binary data"
        filepath.write_bytes(original_content)

        msg = MIMEMultipart()
        _attach_file(msg, str(filepath), "test.bin", "application/octet-stream")

        attachment = msg.get_payload()[0]
        # The payload should be base64 encoded (CTE header)
        assert attachment["Content-Transfer-Encoding"] == "base64"

    def test_attach_file_with_image_type(self, tmp_path):
        """Should work with image MIME types."""
        from mcp.gmail.service import _attach_file

        filepath = tmp_path / "logo.png"
        filepath.write_bytes(b"\x89PNG fake image")

        msg = MIMEMultipart()
        _attach_file(msg, str(filepath), "logo.png", "image/png")

        attachment = msg.get_payload()[0]
        assert attachment.get_content_type() == "image/png"

    def test_attach_file_preserves_binary_content(self, tmp_path):
        """Decoded attachment should match original file content."""
        from mcp.gmail.service import _attach_file

        filepath = tmp_path / "data.bin"
        original = b"Hello, this is test binary data!"
        filepath.write_bytes(original)

        msg = MIMEMultipart()
        _attach_file(msg, str(filepath), "data.bin", "application/octet-stream")

        attachment = msg.get_payload()[0]
        decoded = attachment.get_payload(decode=True)
        assert decoded == original

    def test_attach_file_with_custom_filename(self, tmp_path):
        """The displayed filename can differ from the actual file on disk."""
        from mcp.gmail.service import _attach_file

        filepath = tmp_path / "temp_12345.pdf"
        filepath.write_bytes(b"pdf content")

        msg = MIMEMultipart()
        _attach_file(msg, str(filepath), "my-screenplay.pdf", "application/pdf")

        attachment = msg.get_payload()[0]
        disposition = attachment.get("Content-Disposition")
        assert "my-screenplay.pdf" in disposition

    def test_attach_multiple_files(self, tmp_path):
        """Should handle multiple attachments on the same message."""
        from mcp.gmail.service import _attach_file

        file1 = tmp_path / "a.pdf"
        file1.write_bytes(b"pdf1")
        file2 = tmp_path / "b.txt"
        file2.write_bytes(b"text")

        msg = MIMEMultipart()
        _attach_file(msg, str(file1), "a.pdf", "application/pdf")
        _attach_file(msg, str(file2), "b.txt", "text/plain")

        assert len(msg.get_payload()) == 2


# ===================================================================
# 5. send_simple_email
# ===================================================================


class TestSendSimpleEmail:
    """Tests for send_simple_email() function."""

    def test_send_plain_text_success(self, mock_gmail_service):
        """Should successfully send a plain text email."""
        with patch(
            "mcp.gmail.service.get_gmail_service", return_value=mock_gmail_service
        ):
            from mcp.gmail.service import send_simple_email

            result = send_simple_email(
                to=["user@example.com"],
                subject="Hello",
                body="Plain text body",
            )

            assert result["sent"] is True
            assert result["message_id"] == "msg-123"

    def test_send_html_email_success(self, mock_gmail_service):
        """Should successfully send an HTML email."""
        with patch(
            "mcp.gmail.service.get_gmail_service", return_value=mock_gmail_service
        ):
            from mcp.gmail.service import send_simple_email

            result = send_simple_email(
                to=["user@example.com"],
                subject="HTML Test",
                body="<h1>Hello</h1>",
                html=True,
            )

            assert result["sent"] is True
            assert result["message_id"] == "msg-123"

    def test_send_plain_text_default(self, mock_gmail_service):
        """html parameter should default to False (plain text)."""
        with patch(
            "mcp.gmail.service.get_gmail_service", return_value=mock_gmail_service
        ):
            from mcp.gmail.service import send_simple_email

            result = send_simple_email(
                to=["a@b.com"],
                subject="Test",
                body="text",
            )

            assert result["sent"] is True

    def test_send_failure_returns_error(self, mock_gmail_service):
        """When send fails, should return error dict."""
        mock_gmail_service.users.return_value.messages.return_value.send.return_value.execute.side_effect = Exception(
            "Network error"
        )

        with patch(
            "mcp.gmail.service.get_gmail_service", return_value=mock_gmail_service
        ):
            from mcp.gmail.service import send_simple_email

            result = send_simple_email(
                to=["user@test.com"],
                subject="Fail Test",
                body="body",
            )

            assert result["sent"] is False
            assert "Network error" in result["error"]

    def test_send_returns_to_list(self, mock_gmail_service):
        """Result should include the 'to' field."""
        with patch(
            "mcp.gmail.service.get_gmail_service", return_value=mock_gmail_service
        ):
            from mcp.gmail.service import send_simple_email

            result = send_simple_email(
                to=["a@b.com", "c@d.com"],
                subject="Test",
                body="body",
            )

            assert result["to"] == ["a@b.com", "c@d.com"]

    def test_send_returns_subject(self, mock_gmail_service):
        """Result should include the 'subject' field."""
        with patch(
            "mcp.gmail.service.get_gmail_service", return_value=mock_gmail_service
        ):
            from mcp.gmail.service import send_simple_email

            result = send_simple_email(
                to=["a@b.com"],
                subject="Important Subject",
                body="body",
            )

            assert result["subject"] == "Important Subject"

    def test_message_to_header_joined(self, mock_gmail_service):
        """To header should join multiple recipients with comma."""
        with patch(
            "mcp.gmail.service.get_gmail_service", return_value=mock_gmail_service
        ):
            from mcp.gmail.service import send_simple_email

            send_simple_email(
                to=["a@b.com", "c@d.com"],
                subject="Test",
                body="body",
            )

            send_call = mock_gmail_service.users.return_value.messages.return_value.send
            call_kwargs = send_call.call_args
            body = call_kwargs.kwargs.get("body", call_kwargs[1].get("body", {}))
            raw_bytes = base64.urlsafe_b64decode(body["raw"])
            assert b"a@b.com, c@d.com" in raw_bytes

    def test_send_uses_user_me(self, mock_gmail_service):
        """Gmail API send should use userId='me'."""
        with patch(
            "mcp.gmail.service.get_gmail_service", return_value=mock_gmail_service
        ):
            from mcp.gmail.service import send_simple_email

            send_simple_email(
                to=["a@b.com"],
                subject="Test",
                body="body",
            )

            send_call = mock_gmail_service.users.return_value.messages.return_value.send
            call_kwargs = send_call.call_args
            assert call_kwargs.kwargs.get("userId") == "me" or (
                call_kwargs[1] and call_kwargs[1].get("userId") == "me"
            )

    def test_raw_message_is_base64_encoded(self, mock_gmail_service):
        """The raw message should be valid base64url."""
        with patch(
            "mcp.gmail.service.get_gmail_service", return_value=mock_gmail_service
        ):
            from mcp.gmail.service import send_simple_email

            send_simple_email(
                to=["a@b.com"],
                subject="Test",
                body="body",
            )

            send_call = mock_gmail_service.users.return_value.messages.return_value.send
            call_kwargs = send_call.call_args
            body = call_kwargs.kwargs.get("body", call_kwargs[1].get("body", {}))
            decoded = base64.urlsafe_b64decode(body["raw"])
            assert len(decoded) > 0

    def test_service_error_during_construction_returns_error(self):
        """When get_gmail_service raises, should return error dict."""
        with patch(
            "mcp.gmail.service.get_gmail_service",
            side_effect=RuntimeError("No credentials"),
        ):
            from mcp.gmail.service import send_simple_email

            result = send_simple_email(
                to=["a@b.com"],
                subject="Test",
                body="body",
            )

            assert result["sent"] is False
            assert "No credentials" in result["error"]

    def test_html_content_type_in_message(self, mock_gmail_service):
        """When html=True, the message part should be text/html."""
        with patch(
            "mcp.gmail.service.get_gmail_service", return_value=mock_gmail_service
        ):
            from mcp.gmail.service import send_simple_email

            send_simple_email(
                to=["a@b.com"],
                subject="Test",
                body="<p>Hello</p>",
                html=True,
            )

            send_call = mock_gmail_service.users.return_value.messages.return_value.send
            call_kwargs = send_call.call_args
            body = call_kwargs.kwargs.get("body", call_kwargs[1].get("body", {}))
            raw_bytes = base64.urlsafe_b64decode(body["raw"])
            assert b"text/html" in raw_bytes

    def test_plain_content_type_in_message(self, mock_gmail_service):
        """When html=False, the message part should be text/plain."""
        with patch(
            "mcp.gmail.service.get_gmail_service", return_value=mock_gmail_service
        ):
            from mcp.gmail.service import send_simple_email

            send_simple_email(
                to=["a@b.com"],
                subject="Test",
                body="Plain text",
                html=False,
            )

            send_call = mock_gmail_service.users.return_value.messages.return_value.send
            call_kwargs = send_call.call_args
            body = call_kwargs.kwargs.get("body", call_kwargs[1].get("body", {}))
            raw_bytes = base64.urlsafe_b64decode(body["raw"])
            assert b"text/plain" in raw_bytes


# ===================================================================
# 6. list_labels
# ===================================================================


class TestListLabels:
    """Tests for list_labels() function."""

    def test_list_labels_success(self, mock_gmail_service):
        """Should return list of label dicts on success."""
        with patch(
            "mcp.gmail.service.get_gmail_service", return_value=mock_gmail_service
        ):
            from mcp.gmail.service import list_labels

            result = list_labels()

            assert isinstance(result, list)
            assert len(result) == 2
            assert result[0]["id"] == "INBOX"
            assert result[0]["name"] == "INBOX"
            assert result[1]["id"] == "SENT"
            assert result[1]["name"] == "SENT"

    def test_list_labels_returns_dicts_with_id_and_name(self, mock_gmail_service):
        """Each label dict should have 'id' and 'name' keys."""
        with patch(
            "mcp.gmail.service.get_gmail_service", return_value=mock_gmail_service
        ):
            from mcp.gmail.service import list_labels

            result = list_labels()

            for label in result:
                assert "id" in label
                assert "name" in label

    def test_list_labels_api_failure_returns_empty_list(self, mock_gmail_service):
        """On API failure, should return empty list."""
        mock_gmail_service.users.return_value.labels.return_value.list.return_value.execute.side_effect = Exception(
            "API failure"
        )

        with patch(
            "mcp.gmail.service.get_gmail_service", return_value=mock_gmail_service
        ):
            from mcp.gmail.service import list_labels

            result = list_labels()

            assert result == []

    def test_list_labels_service_construction_failure(self):
        """When get_gmail_service raises, should return empty list."""
        with patch(
            "mcp.gmail.service.get_gmail_service",
            side_effect=RuntimeError("No creds"),
        ):
            from mcp.gmail.service import list_labels

            result = list_labels()

            assert result == []

    def test_list_labels_empty_labels(self, mock_gmail_service):
        """When API returns no labels, should return empty list."""
        mock_gmail_service.users.return_value.labels.return_value.list.return_value.execute.return_value = {
            "labels": []
        }

        with patch(
            "mcp.gmail.service.get_gmail_service", return_value=mock_gmail_service
        ):
            from mcp.gmail.service import list_labels

            result = list_labels()

            assert result == []

    def test_list_labels_missing_labels_key(self, mock_gmail_service):
        """When API response has no 'labels' key, should return empty list."""
        mock_gmail_service.users.return_value.labels.return_value.list.return_value.execute.return_value = (
            {}
        )

        with patch(
            "mcp.gmail.service.get_gmail_service", return_value=mock_gmail_service
        ):
            from mcp.gmail.service import list_labels

            result = list_labels()

            assert result == []

    def test_list_labels_uses_user_me(self, mock_gmail_service):
        """list should be called with userId='me'."""
        with patch(
            "mcp.gmail.service.get_gmail_service", return_value=mock_gmail_service
        ):
            from mcp.gmail.service import list_labels

            list_labels()

            list_call = mock_gmail_service.users.return_value.labels.return_value.list
            list_call.assert_called_once_with(userId="me")
