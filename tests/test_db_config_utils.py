"""
Tests for db/config.py and db/utils.py
=======================================

Comprehensive tests covering:
- DatabaseSettings: defaults, from_env, frozen immutability, sqlalchemy_url, options
- get_engine: lru_cache behavior, mocked create_engine
- create_all: in-memory SQLite table creation, default engine fallback
- get_schema_snapshot: introspection of created tables

Run with: pytest tests/test_db_config_utils.py -v
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine, Column, Integer, String, Text, inspect
from sqlalchemy.orm import DeclarativeBase

# Ensure repo root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from db.config import DatabaseSettings, get_engine


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture(autouse=True)
def clear_engine_cache():
    """Clear the lru_cache on get_engine before and after every test."""
    get_engine.cache_clear()
    yield
    get_engine.cache_clear()


@pytest.fixture()
def clean_env(monkeypatch):
    """Remove all DB_* env vars so from_env uses pure defaults."""
    for var in (
        "DB_HOST",
        "DB_PORT",
        "DB_NAME",
        "DB_USER",
        "DB_PASSWORD",
        "DB_SSLMODE",
    ):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture()
def default_settings():
    """Return a DatabaseSettings with all default values."""
    return DatabaseSettings(
        host="localhost",
        port=5432,
        name="vectordb",
        user="vectoruser",
        password="friday",
        options={},
    )


@pytest.fixture()
def settings_with_ssl():
    """Return a DatabaseSettings with sslmode option."""
    return DatabaseSettings(
        host="db.example.com",
        port=5433,
        name="proddb",
        user="admin",
        password="secret",
        options={"sslmode": "require"},
    )


@pytest.fixture()
def settings_with_multiple_options():
    """Return a DatabaseSettings with multiple options."""
    return DatabaseSettings(
        host="db.example.com",
        port=5433,
        name="proddb",
        user="admin",
        password="secret",
        options={"sslmode": "verify-full", "connect_timeout": "10"},
    )


@pytest.fixture()
def sqlite_engine():
    """Provide an in-memory SQLite engine for table creation tests."""
    engine = create_engine("sqlite://", echo=False)
    yield engine
    engine.dispose()


class _TestBase(DeclarativeBase):
    """A minimal ORM base for testing create_all / get_schema_snapshot."""

    pass


class _Users(_TestBase):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(64), nullable=False)
    email = Column(String(128))


class _Posts(_TestBase):
    __tablename__ = "posts"
    id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String(256), nullable=False)
    body = Column(Text)
    author_id = Column(Integer)


@pytest.fixture()
def sqlite_engine_with_tables():
    """Provide a SQLite engine with _Users and _Posts tables already created."""
    engine = create_engine("sqlite://", echo=False)
    _TestBase.metadata.create_all(engine)
    yield engine
    engine.dispose()


# ============================================================================
# 1. DatabaseSettings -- Construction and field access
# ============================================================================


class TestDatabaseSettingsConstruction:
    """Test direct construction of DatabaseSettings."""

    def test_create_with_positional_args(self):
        ds = DatabaseSettings("myhost", 1234, "mydb", "myuser", "mypass", {})
        assert ds.host == "myhost"
        assert ds.port == 1234
        assert ds.name == "mydb"
        assert ds.user == "myuser"
        assert ds.password == "mypass"
        assert ds.options == {}

    def test_create_with_keyword_args(self):
        ds = DatabaseSettings(
            host="h", port=99, name="n", user="u", password="p", options={"a": "b"}
        )
        assert ds.host == "h"
        assert ds.port == 99
        assert ds.options == {"a": "b"}

    def test_field_types(self, default_settings):
        assert isinstance(default_settings.host, str)
        assert isinstance(default_settings.port, int)
        assert isinstance(default_settings.name, str)
        assert isinstance(default_settings.user, str)
        assert isinstance(default_settings.password, str)
        assert isinstance(default_settings.options, dict)

    def test_options_dict_can_hold_any_type(self):
        ds = DatabaseSettings(
            "h",
            1,
            "n",
            "u",
            "p",
            {"str_val": "hello", "int_val": 42, "bool_val": True, "list_val": [1, 2]},
        )
        assert ds.options["str_val"] == "hello"
        assert ds.options["int_val"] == 42
        assert ds.options["bool_val"] is True
        assert ds.options["list_val"] == [1, 2]

    def test_empty_string_fields_allowed(self):
        ds = DatabaseSettings("", 0, "", "", "", {})
        assert ds.host == ""
        assert ds.port == 0
        assert ds.name == ""

    def test_special_characters_in_password(self):
        ds = DatabaseSettings("h", 1, "n", "u", "p@ss!w0rd#$%", {})
        assert ds.password == "p@ss!w0rd#$%"

    def test_options_empty_dict_by_default_from_env(self, clean_env):
        ds = DatabaseSettings.from_env()
        assert ds.options == {}


# ============================================================================
# 2. DatabaseSettings -- Frozen (immutability)
# ============================================================================


class TestDatabaseSettingsFrozen:
    """Verify that DatabaseSettings is truly frozen (immutable)."""

    def test_cannot_set_host(self, default_settings):
        with pytest.raises(dataclasses.FrozenInstanceError):
            default_settings.host = "newhost"

    def test_cannot_set_port(self, default_settings):
        with pytest.raises(dataclasses.FrozenInstanceError):
            default_settings.port = 9999

    def test_cannot_set_name(self, default_settings):
        with pytest.raises(dataclasses.FrozenInstanceError):
            default_settings.name = "newdb"

    def test_cannot_set_user(self, default_settings):
        with pytest.raises(dataclasses.FrozenInstanceError):
            default_settings.user = "newuser"

    def test_cannot_set_password(self, default_settings):
        with pytest.raises(dataclasses.FrozenInstanceError):
            default_settings.password = "newpass"

    def test_cannot_set_options(self, default_settings):
        with pytest.raises(dataclasses.FrozenInstanceError):
            default_settings.options = {"new": "option"}

    def test_cannot_delete_field(self, default_settings):
        with pytest.raises(dataclasses.FrozenInstanceError):
            del default_settings.host

    def test_cannot_delete_port(self, default_settings):
        with pytest.raises(dataclasses.FrozenInstanceError):
            del default_settings.port

    def test_cannot_delete_name(self, default_settings):
        with pytest.raises(dataclasses.FrozenInstanceError):
            del default_settings.name

    def test_cannot_add_new_attribute(self, default_settings):
        with pytest.raises(dataclasses.FrozenInstanceError):
            default_settings.new_attr = "value"

    def test_is_dataclass(self):
        assert dataclasses.is_dataclass(DatabaseSettings)

    def test_frozen_flag_is_true(self):
        """Verify frozen via attempted mutation on a fresh instance."""
        ds = DatabaseSettings("h", 1, "n", "u", "p", {})
        with pytest.raises(dataclasses.FrozenInstanceError):
            ds.host = "x"

    def test_options_dict_contents_still_mutable(self, default_settings):
        """The dict itself is mutable (Python limitation), but the field ref is frozen."""
        # The options dict can be mutated in-place (not a deep freeze)
        default_settings.options["injected"] = "value"
        assert default_settings.options["injected"] == "value"


# ============================================================================
# 3. DatabaseSettings -- Equality
# ============================================================================


class TestDatabaseSettingsEquality:
    """Test equality comparison on DatabaseSettings."""

    def test_equal_instances(self):
        ds1 = DatabaseSettings("h", 1, "n", "u", "p", {})
        ds2 = DatabaseSettings("h", 1, "n", "u", "p", {})
        assert ds1 == ds2

    def test_unequal_instances_different_host(self):
        ds1 = DatabaseSettings("h1", 1, "n", "u", "p", {})
        ds2 = DatabaseSettings("h2", 1, "n", "u", "p", {})
        assert ds1 != ds2

    def test_unequal_instances_different_port(self):
        ds1 = DatabaseSettings("h", 1, "n", "u", "p", {})
        ds2 = DatabaseSettings("h", 2, "n", "u", "p", {})
        assert ds1 != ds2

    def test_unequal_instances_different_name(self):
        ds1 = DatabaseSettings("h", 1, "n1", "u", "p", {})
        ds2 = DatabaseSettings("h", 1, "n2", "u", "p", {})
        assert ds1 != ds2

    def test_unequal_instances_different_user(self):
        ds1 = DatabaseSettings("h", 1, "n", "u1", "p", {})
        ds2 = DatabaseSettings("h", 1, "n", "u2", "p", {})
        assert ds1 != ds2

    def test_unequal_instances_different_password(self):
        ds1 = DatabaseSettings("h", 1, "n", "u", "p1", {})
        ds2 = DatabaseSettings("h", 1, "n", "u", "p2", {})
        assert ds1 != ds2

    def test_unequal_instances_different_options(self):
        ds1 = DatabaseSettings("h", 1, "n", "u", "p", {})
        ds2 = DatabaseSettings("h", 1, "n", "u", "p", {"sslmode": "require"})
        assert ds1 != ds2

    def test_not_equal_to_non_settings(self, default_settings):
        assert default_settings != "not a settings object"
        assert default_settings != 42
        assert default_settings != None

    def test_not_equal_to_tuple(self):
        ds = DatabaseSettings("h", 1, "n", "u", "p", {})
        assert ds != ("h", 1, "n", "u", "p", {})

    def test_equal_instances_with_same_options(self):
        ds1 = DatabaseSettings("h", 1, "n", "u", "p", {"ssl": "on"})
        ds2 = DatabaseSettings("h", 1, "n", "u", "p", {"ssl": "on"})
        assert ds1 == ds2


# ============================================================================
# 4. DatabaseSettings -- Hashing behavior
# ============================================================================


class TestDatabaseSettingsHashing:
    """Frozen dataclass with dict field: hash raises TypeError."""

    def test_hash_raises_with_dict_options(self):
        """A dict field makes the frozen dataclass unhashable."""
        ds = DatabaseSettings("h", 1, "n", "u", "p", {})
        with pytest.raises(TypeError, match="unhashable type"):
            hash(ds)

    def test_hash_raises_with_nonempty_dict(self):
        ds = DatabaseSettings("h", 1, "n", "u", "p", {"ssl": "on"})
        with pytest.raises(TypeError, match="unhashable type"):
            hash(ds)

    def test_cannot_use_as_set_element(self):
        ds = DatabaseSettings("h", 1, "n", "u", "p", {})
        with pytest.raises(TypeError):
            {ds}

    def test_cannot_use_as_dict_key(self):
        ds = DatabaseSettings("h", 1, "n", "u", "p", {})
        with pytest.raises(TypeError):
            {ds: "value"}


# ============================================================================
# 5. DatabaseSettings.from_env -- Default environment
# ============================================================================


class TestFromEnvDefaults:
    """Test from_env when no env vars are set (all defaults)."""

    def test_default_host(self, clean_env):
        ds = DatabaseSettings.from_env()
        assert ds.host == "localhost"

    def test_default_port(self, clean_env):
        ds = DatabaseSettings.from_env()
        assert ds.port == 5432

    def test_default_name(self, clean_env):
        ds = DatabaseSettings.from_env()
        assert ds.name == "vectordb"

    def test_default_user(self, clean_env):
        ds = DatabaseSettings.from_env()
        assert ds.user == "vectoruser"

    def test_default_password(self, clean_env):
        ds = DatabaseSettings.from_env()
        assert ds.password == "friday"

    def test_default_options_empty(self, clean_env):
        ds = DatabaseSettings.from_env()
        assert ds.options == {}

    def test_default_port_is_int(self, clean_env):
        ds = DatabaseSettings.from_env()
        assert isinstance(ds.port, int)

    def test_all_defaults_together(self, clean_env):
        ds = DatabaseSettings.from_env()
        expected = DatabaseSettings(
            "localhost", 5432, "vectordb", "vectoruser", "friday", {}
        )
        assert ds == expected

    def test_default_options_is_dict(self, clean_env):
        ds = DatabaseSettings.from_env()
        assert isinstance(ds.options, dict)

    def test_default_options_has_no_sslmode(self, clean_env):
        ds = DatabaseSettings.from_env()
        assert "sslmode" not in ds.options


# ============================================================================
# 6. DatabaseSettings.from_env -- Custom environment variables
# ============================================================================


class TestFromEnvCustom:
    """Test from_env with custom env vars via monkeypatch."""

    def test_custom_host(self, monkeypatch, clean_env):
        monkeypatch.setenv("DB_HOST", "db.production.com")
        ds = DatabaseSettings.from_env()
        assert ds.host == "db.production.com"

    def test_custom_port(self, monkeypatch, clean_env):
        monkeypatch.setenv("DB_PORT", "5433")
        ds = DatabaseSettings.from_env()
        assert ds.port == 5433

    def test_custom_port_is_int(self, monkeypatch, clean_env):
        monkeypatch.setenv("DB_PORT", "9999")
        ds = DatabaseSettings.from_env()
        assert isinstance(ds.port, int)
        assert ds.port == 9999

    def test_custom_name(self, monkeypatch, clean_env):
        monkeypatch.setenv("DB_NAME", "production_db")
        ds = DatabaseSettings.from_env()
        assert ds.name == "production_db"

    def test_custom_user(self, monkeypatch, clean_env):
        monkeypatch.setenv("DB_USER", "prodadmin")
        ds = DatabaseSettings.from_env()
        assert ds.user == "prodadmin"

    def test_custom_password(self, monkeypatch, clean_env):
        monkeypatch.setenv("DB_PASSWORD", "super_secret_123!")
        ds = DatabaseSettings.from_env()
        assert ds.password == "super_secret_123!"

    def test_all_custom_env_vars(self, monkeypatch, clean_env):
        monkeypatch.setenv("DB_HOST", "custom-host")
        monkeypatch.setenv("DB_PORT", "1111")
        monkeypatch.setenv("DB_NAME", "custom-db")
        monkeypatch.setenv("DB_USER", "custom-user")
        monkeypatch.setenv("DB_PASSWORD", "custom-pass")
        ds = DatabaseSettings.from_env()
        assert ds.host == "custom-host"
        assert ds.port == 1111
        assert ds.name == "custom-db"
        assert ds.user == "custom-user"
        assert ds.password == "custom-pass"
        assert ds.options == {}

    def test_sslmode_set(self, monkeypatch, clean_env):
        monkeypatch.setenv("DB_SSLMODE", "require")
        ds = DatabaseSettings.from_env()
        assert ds.options == {"sslmode": "require"}

    def test_sslmode_verify_full(self, monkeypatch, clean_env):
        monkeypatch.setenv("DB_SSLMODE", "verify-full")
        ds = DatabaseSettings.from_env()
        assert ds.options["sslmode"] == "verify-full"

    def test_sslmode_verify_ca(self, monkeypatch, clean_env):
        monkeypatch.setenv("DB_SSLMODE", "verify-ca")
        ds = DatabaseSettings.from_env()
        assert ds.options["sslmode"] == "verify-ca"

    def test_sslmode_disable(self, monkeypatch, clean_env):
        monkeypatch.setenv("DB_SSLMODE", "disable")
        ds = DatabaseSettings.from_env()
        assert ds.options["sslmode"] == "disable"

    def test_sslmode_prefer(self, monkeypatch, clean_env):
        monkeypatch.setenv("DB_SSLMODE", "prefer")
        ds = DatabaseSettings.from_env()
        assert ds.options["sslmode"] == "prefer"

    def test_sslmode_allow(self, monkeypatch, clean_env):
        monkeypatch.setenv("DB_SSLMODE", "allow")
        ds = DatabaseSettings.from_env()
        assert ds.options["sslmode"] == "allow"

    def test_no_sslmode_env_var(self, clean_env):
        ds = DatabaseSettings.from_env()
        assert "sslmode" not in ds.options

    def test_sslmode_empty_string_is_falsy(self, monkeypatch, clean_env):
        monkeypatch.setenv("DB_SSLMODE", "")
        ds = DatabaseSettings.from_env()
        # Empty string is falsy, so sslmode should not be in options
        assert "sslmode" not in ds.options

    def test_port_invalid_raises_value_error(self, monkeypatch, clean_env):
        monkeypatch.setenv("DB_PORT", "not_a_number")
        with pytest.raises(ValueError):
            DatabaseSettings.from_env()

    def test_port_float_string_raises_value_error(self, monkeypatch, clean_env):
        monkeypatch.setenv("DB_PORT", "54.32")
        with pytest.raises(ValueError):
            DatabaseSettings.from_env()

    def test_port_empty_string_raises_value_error(self, monkeypatch, clean_env):
        monkeypatch.setenv("DB_PORT", "")
        with pytest.raises(ValueError):
            DatabaseSettings.from_env()

    def test_returns_database_settings_instance(self, clean_env):
        ds = DatabaseSettings.from_env()
        assert isinstance(ds, DatabaseSettings)

    def test_from_env_is_classmethod(self):
        assert isinstance(DatabaseSettings.__dict__["from_env"], classmethod)

    def test_sslmode_only_option_key(self, monkeypatch, clean_env):
        """Only sslmode is read from env into options; no other keys."""
        monkeypatch.setenv("DB_SSLMODE", "require")
        ds = DatabaseSettings.from_env()
        assert list(ds.options.keys()) == ["sslmode"]

    def test_custom_host_preserves_other_defaults(self, monkeypatch, clean_env):
        monkeypatch.setenv("DB_HOST", "custom-only-host")
        ds = DatabaseSettings.from_env()
        assert ds.host == "custom-only-host"
        assert ds.port == 5432
        assert ds.name == "vectordb"
        assert ds.user == "vectoruser"
        assert ds.password == "friday"


# ============================================================================
# 7. DatabaseSettings.sqlalchemy_url -- URL generation
# ============================================================================


class TestSqlalchemyUrl:
    """Test sqlalchemy_url() method on DatabaseSettings."""

    def test_url_with_defaults(self, default_settings):
        url = default_settings.sqlalchemy_url()
        assert url == "postgresql+psycopg://vectoruser:friday@localhost:5432/vectordb"

    def test_url_format_structure(self, default_settings):
        url = default_settings.sqlalchemy_url()
        assert url.startswith("postgresql+psycopg://")
        assert "@" in url
        assert ":" in url

    def test_url_contains_host(self):
        ds = DatabaseSettings("myhost.example.com", 5432, "db", "u", "p", {})
        url = ds.sqlalchemy_url()
        assert "myhost.example.com" in url

    def test_url_contains_port(self):
        ds = DatabaseSettings("h", 9999, "db", "u", "p", {})
        url = ds.sqlalchemy_url()
        assert ":9999/" in url

    def test_url_contains_dbname(self):
        ds = DatabaseSettings("h", 1, "mydatabase", "u", "p", {})
        url = ds.sqlalchemy_url()
        assert url.endswith("/mydatabase")

    def test_url_contains_user(self):
        ds = DatabaseSettings("h", 1, "db", "myuser", "p", {})
        url = ds.sqlalchemy_url()
        assert "myuser:" in url

    def test_url_contains_password(self):
        ds = DatabaseSettings("h", 1, "db", "u", "mypassword", {})
        url = ds.sqlalchemy_url()
        assert ":mypassword@" in url

    def test_url_no_options(self, default_settings):
        url = default_settings.sqlalchemy_url()
        assert "?" not in url

    def test_url_with_single_option(self, settings_with_ssl):
        url = settings_with_ssl.sqlalchemy_url()
        assert "?sslmode=require" in url

    def test_url_with_ssl_full(self, settings_with_ssl):
        url = settings_with_ssl.sqlalchemy_url()
        expected = "postgresql+psycopg://admin:secret@db.example.com:5433/proddb?sslmode=require"
        assert url == expected

    def test_url_with_multiple_options(self, settings_with_multiple_options):
        url = settings_with_multiple_options.sqlalchemy_url()
        assert "?" in url
        # Check both options are present (order depends on dict iteration)
        assert "sslmode=verify-full" in url
        assert "connect_timeout=10" in url
        # They should be joined by &
        assert "&" in url

    def test_url_options_delimiter_ampersand(self):
        ds = DatabaseSettings("h", 1, "db", "u", "p", {"a": "1", "b": "2", "c": "3"})
        url = ds.sqlalchemy_url()
        query_string = url.split("?")[1]
        parts = query_string.split("&")
        assert len(parts) == 3

    def test_url_returns_string(self, default_settings):
        url = default_settings.sqlalchemy_url()
        assert isinstance(url, str)

    def test_url_with_special_chars_in_password(self):
        ds = DatabaseSettings("h", 1, "db", "u", "p@ss:word", {})
        url = ds.sqlalchemy_url()
        # The password is inserted as-is (no URL encoding in this implementation)
        assert "p@ss:word" in url

    def test_url_with_empty_options_dict(self):
        ds = DatabaseSettings("h", 1, "db", "u", "p", {})
        url = ds.sqlalchemy_url()
        assert "?" not in url

    def test_url_with_numeric_option_value(self):
        ds = DatabaseSettings("h", 1, "db", "u", "p", {"timeout": 30})
        url = ds.sqlalchemy_url()
        assert "?timeout=30" in url

    def test_url_with_boolean_option_value(self):
        ds = DatabaseSettings("h", 1, "db", "u", "p", {"sslmode": True})
        url = ds.sqlalchemy_url()
        assert "?sslmode=True" in url

    def test_url_driver_prefix(self, default_settings):
        url = default_settings.sqlalchemy_url()
        assert url.startswith("postgresql+psycopg://")

    def test_url_host_port_separator(self):
        ds = DatabaseSettings("myhost", 5432, "db", "u", "p", {})
        url = ds.sqlalchemy_url()
        assert "@myhost:5432/" in url

    def test_url_with_single_option_has_one_question_mark(self):
        ds = DatabaseSettings("h", 1, "db", "u", "p", {"key": "val"})
        url = ds.sqlalchemy_url()
        assert url.count("?") == 1

    def test_url_with_multiple_options_has_one_question_mark(self):
        ds = DatabaseSettings("h", 1, "db", "u", "p", {"a": "1", "b": "2"})
        url = ds.sqlalchemy_url()
        assert url.count("?") == 1


# ============================================================================
# 8. DatabaseSettings -- Dataclass fields
# ============================================================================


class TestDatabaseSettingsFields:
    """Test the dataclass fields metadata."""

    def test_field_count(self):
        fields = dataclasses.fields(DatabaseSettings)
        assert len(fields) == 6

    def test_field_names(self):
        names = [f.name for f in dataclasses.fields(DatabaseSettings)]
        assert names == ["host", "port", "name", "user", "password", "options"]

    def test_field_types(self):
        fields = {f.name: f.type for f in dataclasses.fields(DatabaseSettings)}
        assert fields["host"] == "str"
        assert fields["port"] == "int"
        assert fields["name"] == "str"
        assert fields["user"] == "str"
        assert fields["password"] == "str"

    def test_repr(self, default_settings):
        r = repr(default_settings)
        assert "DatabaseSettings" in r
        assert "localhost" in r
        assert "5432" in r
        assert "vectordb" in r

    def test_repr_contains_all_fields(self):
        ds = DatabaseSettings("myhost", 9999, "mydb", "myuser", "mypass", {"ssl": "on"})
        r = repr(ds)
        assert "myhost" in r
        assert "9999" in r
        assert "mydb" in r
        assert "myuser" in r
        assert "mypass" in r
        assert "ssl" in r


# ============================================================================
# 9. get_engine -- Mocked create_engine (using None/default arg)
# ============================================================================


class TestGetEngine:
    """Test get_engine function with mocked create_engine to avoid real DB.

    Note: DatabaseSettings contains a dict field which makes it unhashable.
    The lru_cache on get_engine works because the typical call path is
    get_engine(None) or get_engine() -- None is hashable. Tests that need to
    pass explicit settings must mock DatabaseSettings.from_env instead.
    """

    @patch("db.config.create_engine")
    def test_returns_engine_with_none(self, mock_create_engine, clean_env):
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine
        engine = get_engine(None)
        assert engine is mock_engine

    @patch("db.config.create_engine")
    def test_returns_engine_default_arg(self, mock_create_engine, clean_env):
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine
        engine = get_engine()
        assert engine is mock_engine

    @patch("db.config.create_engine")
    def test_calls_create_engine_with_default_url(self, mock_create_engine, clean_env):
        mock_create_engine.return_value = MagicMock()
        get_engine()
        expected_url = "postgresql+psycopg://vectoruser:friday@localhost:5432/vectordb"
        mock_create_engine.assert_called_once_with(expected_url, future=True)

    @patch("db.config.create_engine")
    def test_calls_create_engine_with_future_true(self, mock_create_engine, clean_env):
        mock_create_engine.return_value = MagicMock()
        get_engine()
        _, kwargs = mock_create_engine.call_args
        assert kwargs["future"] is True

    @patch("db.config.create_engine")
    def test_uses_from_env_when_settings_none(self, mock_create_engine, clean_env):
        mock_create_engine.return_value = MagicMock()
        get_engine(None)
        # from_env is called internally, resulting in the default URL
        url_arg = mock_create_engine.call_args[0][0]
        assert "localhost" in url_arg
        assert "vectordb" in url_arg

    @patch("db.config.create_engine")
    def test_custom_env_reflected_in_url(
        self, mock_create_engine, monkeypatch, clean_env
    ):
        """get_engine() with custom env vars produces the correct URL."""
        monkeypatch.setenv("DB_HOST", "prodhost")
        monkeypatch.setenv("DB_PORT", "5433")
        monkeypatch.setenv("DB_NAME", "proddb")
        monkeypatch.setenv("DB_USER", "admin")
        monkeypatch.setenv("DB_PASSWORD", "secret")
        mock_create_engine.return_value = MagicMock()
        get_engine()
        expected_url = "postgresql+psycopg://admin:secret@prodhost:5433/proddb"
        mock_create_engine.assert_called_once_with(expected_url, future=True)

    @patch("db.config.create_engine")
    def test_sslmode_env_in_url(self, mock_create_engine, monkeypatch, clean_env):
        monkeypatch.setenv("DB_SSLMODE", "verify-full")
        mock_create_engine.return_value = MagicMock()
        get_engine()
        url_arg = mock_create_engine.call_args[0][0]
        assert "sslmode=verify-full" in url_arg

    @patch("db.config.create_engine")
    def test_no_sslmode_no_query_params(self, mock_create_engine, clean_env):
        mock_create_engine.return_value = MagicMock()
        get_engine()
        url_arg = mock_create_engine.call_args[0][0]
        assert "?" not in url_arg

    @patch("db.config.DatabaseSettings.from_env")
    @patch("db.config.create_engine")
    def test_from_env_called_when_none(self, mock_create_engine, mock_from_env):
        """Explicitly verify from_env is called when settings is None."""
        mock_settings = MagicMock()
        mock_settings.sqlalchemy_url.return_value = "postgresql+psycopg://u:p@h:1/d"
        mock_from_env.return_value = mock_settings
        mock_create_engine.return_value = MagicMock()
        get_engine(None)
        mock_from_env.assert_called_once()


# ============================================================================
# 10. get_engine -- LRU cache behavior
# ============================================================================


class TestGetEngineLruCache:
    """Test that get_engine uses lru_cache correctly.

    We test caching with settings=None (the standard call path) since
    DatabaseSettings with a dict field is unhashable and would fail lru_cache.
    """

    @patch("db.config.create_engine")
    def test_cache_returns_same_engine(self, mock_create_engine, clean_env):
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine
        engine1 = get_engine()
        engine2 = get_engine()
        assert engine1 is engine2

    @patch("db.config.create_engine")
    def test_cache_calls_create_engine_only_once(self, mock_create_engine, clean_env):
        mock_create_engine.return_value = MagicMock()
        get_engine()
        get_engine()
        get_engine()
        assert mock_create_engine.call_count == 1

    @patch("db.config.create_engine")
    def test_cache_clear_allows_new_call(self, mock_create_engine, clean_env):
        mock_create_engine.return_value = MagicMock()
        get_engine()
        assert mock_create_engine.call_count == 1
        get_engine.cache_clear()
        get_engine()
        assert mock_create_engine.call_count == 2

    @patch("db.config.create_engine")
    def test_cache_clear_resets_info(self, mock_create_engine, clean_env):
        mock_create_engine.return_value = MagicMock()
        get_engine()
        get_engine.cache_clear()
        info = get_engine.cache_info()
        assert info.hits == 0
        assert info.misses == 0

    @patch("db.config.create_engine")
    def test_cache_info_hits(self, mock_create_engine, clean_env):
        mock_create_engine.return_value = MagicMock()
        get_engine()
        get_engine()
        info = get_engine.cache_info()
        assert info.hits == 1
        assert info.misses == 1

    @patch("db.config.create_engine")
    def test_cache_info_misses(self, mock_create_engine, clean_env):
        mock_create_engine.return_value = MagicMock()
        get_engine()
        info = get_engine.cache_info()
        assert info.misses == 1
        assert info.hits == 0

    @patch("db.config.create_engine")
    def test_cache_maxsize_is_one(self, mock_create_engine, clean_env):
        """lru_cache maxsize=1 is reported correctly."""
        mock_create_engine.return_value = MagicMock()
        info = get_engine.cache_info()
        assert info.maxsize == 1

    @patch("db.config.create_engine")
    def test_multiple_calls_none_cached(self, mock_create_engine, clean_env):
        """Calling with None repeatedly uses cache."""
        mock_create_engine.return_value = MagicMock()
        get_engine(None)
        get_engine(None)
        get_engine(None)
        get_engine(None)
        assert mock_create_engine.call_count == 1
        info = get_engine.cache_info()
        assert info.hits == 3

    def test_cache_clear_method_exists(self):
        assert hasattr(get_engine, "cache_clear")
        assert callable(get_engine.cache_clear)

    def test_cache_info_method_exists(self):
        assert hasattr(get_engine, "cache_info")
        assert callable(get_engine.cache_info)

    @patch("db.config.create_engine")
    def test_cache_after_clear_creates_new_engine(self, mock_create_engine, clean_env):
        engine_a = MagicMock(name="engine_a")
        engine_b = MagicMock(name="engine_b")
        mock_create_engine.side_effect = [engine_a, engine_b]
        result1 = get_engine()
        assert result1 is engine_a
        get_engine.cache_clear()
        result2 = get_engine()
        assert result2 is engine_b
        assert result1 is not result2

    @patch("db.config.create_engine")
    def test_unhashable_settings_raises_type_error(self, mock_create_engine):
        """Passing a DatabaseSettings directly to lru_cached get_engine raises TypeError."""
        mock_create_engine.return_value = MagicMock()
        settings = DatabaseSettings("h", 1, "db", "u", "p", {})
        with pytest.raises(TypeError, match="unhashable type"):
            get_engine(settings)


# ============================================================================
# 11. create_all -- In-memory SQLite
# ============================================================================


class TestCreateAll:
    """Test create_all using in-memory SQLite and the real ORM schema."""

    def test_create_all_with_explicit_engine(self, sqlite_engine):
        from db.utils import create_all

        create_all(engine=sqlite_engine)
        inspector = inspect(sqlite_engine)
        table_names = inspector.get_table_names()
        assert "datasets" in table_names
        assert "model_versions" in table_names

    def test_create_all_creates_training_runs_table(self, sqlite_engine):
        from db.utils import create_all

        create_all(engine=sqlite_engine)
        inspector = inspect(sqlite_engine)
        assert "training_runs" in inspector.get_table_names()

    def test_create_all_creates_eval_suites_table(self, sqlite_engine):
        from db.utils import create_all

        create_all(engine=sqlite_engine)
        inspector = inspect(sqlite_engine)
        assert "eval_suites" in inspector.get_table_names()

    def test_create_all_creates_eval_cases_table(self, sqlite_engine):
        from db.utils import create_all

        create_all(engine=sqlite_engine)
        inspector = inspect(sqlite_engine)
        assert "eval_cases" in inspector.get_table_names()

    def test_create_all_creates_eval_runs_table(self, sqlite_engine):
        from db.utils import create_all

        create_all(engine=sqlite_engine)
        inspector = inspect(sqlite_engine)
        assert "eval_runs" in inspector.get_table_names()

    def test_create_all_creates_eval_results_table(self, sqlite_engine):
        from db.utils import create_all

        create_all(engine=sqlite_engine)
        inspector = inspect(sqlite_engine)
        assert "eval_results" in inspector.get_table_names()

    def test_create_all_creates_artifacts_table(self, sqlite_engine):
        from db.utils import create_all

        create_all(engine=sqlite_engine)
        inspector = inspect(sqlite_engine)
        assert "artifacts" in inspector.get_table_names()

    def test_create_all_idempotent(self, sqlite_engine):
        """Calling create_all twice should not raise."""
        from db.utils import create_all

        create_all(engine=sqlite_engine)
        create_all(engine=sqlite_engine)
        inspector = inspect(sqlite_engine)
        assert "datasets" in inspector.get_table_names()

    @patch("db.utils.get_engine")
    def test_create_all_default_engine(self, mock_get_engine):
        """When no engine is provided, get_engine() is called."""
        from db.utils import create_all

        mock_engine = create_engine("sqlite://", echo=False)
        mock_get_engine.return_value = mock_engine
        create_all()
        mock_get_engine.assert_called_once()
        # Verify tables were created on the mock engine
        inspector = inspect(mock_engine)
        assert "datasets" in inspector.get_table_names()
        mock_engine.dispose()

    def test_create_all_datasets_columns(self, sqlite_engine):
        from db.utils import create_all

        create_all(engine=sqlite_engine)
        inspector = inspect(sqlite_engine)
        columns = [col["name"] for col in inspector.get_columns("datasets")]
        assert "id" in columns
        assert "name" in columns
        assert "dataset_type" in columns
        assert "path" in columns
        assert "examples" in columns
        assert "meta" in columns
        assert "created_at" in columns

    def test_create_all_model_versions_columns(self, sqlite_engine):
        from db.utils import create_all

        create_all(engine=sqlite_engine)
        inspector = inspect(sqlite_engine)
        columns = [col["name"] for col in inspector.get_columns("model_versions")]
        assert "id" in columns
        assert "name" in columns
        assert "base_model" in columns
        assert "adapter_path" in columns
        assert "dataset_id" in columns
        assert "notes" in columns
        assert "created_at" in columns

    def test_create_all_returns_none(self, sqlite_engine):
        from db.utils import create_all

        result = create_all(engine=sqlite_engine)
        assert result is None

    def test_create_all_training_runs_columns(self, sqlite_engine):
        from db.utils import create_all

        create_all(engine=sqlite_engine)
        inspector = inspect(sqlite_engine)
        columns = [col["name"] for col in inspector.get_columns("training_runs")]
        assert "id" in columns
        assert "model_version_id" in columns
        assert "started_at" in columns
        assert "status" in columns
        assert "hyperparams" in columns
        assert "metrics" in columns

    def test_create_all_artifacts_columns(self, sqlite_engine):
        from db.utils import create_all

        create_all(engine=sqlite_engine)
        inspector = inspect(sqlite_engine)
        columns = [col["name"] for col in inspector.get_columns("artifacts")]
        assert "id" in columns
        assert "artifact_type" in columns
        assert "path" in columns
        assert "meta" in columns
        assert "created_at" in columns


# ============================================================================
# 12. get_schema_snapshot -- In-memory SQLite
# ============================================================================


class TestGetSchemaSnapshot:
    """Test get_schema_snapshot with in-memory SQLite."""

    def test_snapshot_empty_db(self):
        """An empty database returns an empty dict."""
        from db.utils import get_schema_snapshot

        engine = create_engine("sqlite://", echo=False)
        snapshot = get_schema_snapshot(engine=engine)
        assert snapshot == {}
        engine.dispose()

    def test_snapshot_with_tables(self, sqlite_engine_with_tables):
        from db.utils import get_schema_snapshot

        snapshot = get_schema_snapshot(engine=sqlite_engine_with_tables)
        assert "users" in snapshot
        assert "posts" in snapshot

    def test_snapshot_users_columns(self, sqlite_engine_with_tables):
        from db.utils import get_schema_snapshot

        snapshot = get_schema_snapshot(engine=sqlite_engine_with_tables)
        assert "id" in snapshot["users"]
        assert "username" in snapshot["users"]
        assert "email" in snapshot["users"]

    def test_snapshot_posts_columns(self, sqlite_engine_with_tables):
        from db.utils import get_schema_snapshot

        snapshot = get_schema_snapshot(engine=sqlite_engine_with_tables)
        assert "id" in snapshot["posts"]
        assert "title" in snapshot["posts"]
        assert "body" in snapshot["posts"]
        assert "author_id" in snapshot["posts"]

    def test_snapshot_returns_dict(self, sqlite_engine_with_tables):
        from db.utils import get_schema_snapshot

        snapshot = get_schema_snapshot(engine=sqlite_engine_with_tables)
        assert isinstance(snapshot, dict)

    def test_snapshot_values_are_lists(self, sqlite_engine_with_tables):
        from db.utils import get_schema_snapshot

        snapshot = get_schema_snapshot(engine=sqlite_engine_with_tables)
        for table_name, columns in snapshot.items():
            assert isinstance(columns, list)

    def test_snapshot_column_names_are_strings(self, sqlite_engine_with_tables):
        from db.utils import get_schema_snapshot

        snapshot = get_schema_snapshot(engine=sqlite_engine_with_tables)
        for table_name, columns in snapshot.items():
            for col in columns:
                assert isinstance(col, str)

    def test_snapshot_table_count(self, sqlite_engine_with_tables):
        from db.utils import get_schema_snapshot

        snapshot = get_schema_snapshot(engine=sqlite_engine_with_tables)
        assert len(snapshot) == 2

    def test_snapshot_users_column_count(self, sqlite_engine_with_tables):
        from db.utils import get_schema_snapshot

        snapshot = get_schema_snapshot(engine=sqlite_engine_with_tables)
        assert len(snapshot["users"]) == 3  # id, username, email

    def test_snapshot_posts_column_count(self, sqlite_engine_with_tables):
        from db.utils import get_schema_snapshot

        snapshot = get_schema_snapshot(engine=sqlite_engine_with_tables)
        assert len(snapshot["posts"]) == 4  # id, title, body, author_id

    @patch("db.utils.get_engine")
    def test_snapshot_default_engine(self, mock_get_engine):
        """When no engine is provided, get_engine() is called."""
        from db.utils import get_schema_snapshot

        mock_engine = create_engine("sqlite://", echo=False)
        _TestBase.metadata.create_all(mock_engine)
        mock_get_engine.return_value = mock_engine
        snapshot = get_schema_snapshot()
        mock_get_engine.assert_called_once()
        assert "users" in snapshot
        mock_engine.dispose()

    def test_snapshot_with_real_schema(self, sqlite_engine):
        """Test snapshot after create_all with the real ORM schema."""
        from db.schema import Base
        from db.utils import get_schema_snapshot

        Base.metadata.create_all(sqlite_engine)
        snapshot = get_schema_snapshot(engine=sqlite_engine)
        assert "datasets" in snapshot
        assert "model_versions" in snapshot
        assert "training_runs" in snapshot
        assert "eval_suites" in snapshot
        assert "eval_cases" in snapshot
        assert "eval_runs" in snapshot
        assert "eval_results" in snapshot
        assert "artifacts" in snapshot

    def test_snapshot_real_schema_datasets_columns(self, sqlite_engine):
        from db.schema import Base
        from db.utils import get_schema_snapshot

        Base.metadata.create_all(sqlite_engine)
        snapshot = get_schema_snapshot(engine=sqlite_engine)
        cols = snapshot["datasets"]
        assert "id" in cols
        assert "name" in cols
        assert "dataset_type" in cols

    def test_snapshot_real_schema_artifacts_columns(self, sqlite_engine):
        from db.schema import Base
        from db.utils import get_schema_snapshot

        Base.metadata.create_all(sqlite_engine)
        snapshot = get_schema_snapshot(engine=sqlite_engine)
        cols = snapshot["artifacts"]
        assert "id" in cols
        assert "artifact_type" in cols
        assert "path" in cols

    def test_snapshot_does_not_include_nonexistent_table(
        self, sqlite_engine_with_tables
    ):
        from db.utils import get_schema_snapshot

        snapshot = get_schema_snapshot(engine=sqlite_engine_with_tables)
        assert "nonexistent_table" not in snapshot

    def test_snapshot_keys_are_strings(self, sqlite_engine_with_tables):
        from db.utils import get_schema_snapshot

        snapshot = get_schema_snapshot(engine=sqlite_engine_with_tables)
        for key in snapshot.keys():
            assert isinstance(key, str)

    def test_snapshot_returns_new_dict_each_call(self, sqlite_engine_with_tables):
        from db.utils import get_schema_snapshot

        snap1 = get_schema_snapshot(engine=sqlite_engine_with_tables)
        snap2 = get_schema_snapshot(engine=sqlite_engine_with_tables)
        assert snap1 == snap2
        assert snap1 is not snap2


# ============================================================================
# 13. create_all and get_schema_snapshot -- Integration
# ============================================================================


class TestCreateAllAndSnapshot:
    """Integration: create tables then snapshot them."""

    def test_create_then_snapshot_round_trip(self):
        """Create all tables, then snapshot and verify consistency."""
        from db.schema import Base
        from db.utils import create_all, get_schema_snapshot

        engine = create_engine("sqlite://", echo=False)
        create_all(engine=engine)
        snapshot = get_schema_snapshot(engine=engine)

        # All tables from Base should be present
        for table_name in Base.metadata.tables:
            assert table_name in snapshot, f"Missing table: {table_name}"

        # Every column in metadata should appear in snapshot
        for table_name, table_obj in Base.metadata.tables.items():
            expected_cols = {col.name for col in table_obj.columns}
            snapshot_cols = set(snapshot[table_name])
            assert expected_cols == snapshot_cols, (
                f"Column mismatch in {table_name}: "
                f"expected {expected_cols}, got {snapshot_cols}"
            )

        engine.dispose()

    def test_snapshot_after_create_has_correct_table_count(self):
        from db.schema import Base
        from db.utils import create_all, get_schema_snapshot

        engine = create_engine("sqlite://", echo=False)
        create_all(engine=engine)
        snapshot = get_schema_snapshot(engine=engine)
        expected_count = len(Base.metadata.tables)
        assert len(snapshot) == expected_count
        engine.dispose()

    def test_snapshot_column_order_matches_schema(self):
        """Snapshot columns should include all columns defined in the ORM."""
        from db.schema import Base
        from db.utils import create_all, get_schema_snapshot

        engine = create_engine("sqlite://", echo=False)
        create_all(engine=engine)
        snapshot = get_schema_snapshot(engine=engine)

        for table_name, table_obj in Base.metadata.tables.items():
            schema_cols = {col.name for col in table_obj.columns}
            snap_cols = set(snapshot[table_name])
            assert schema_cols == snap_cols
        engine.dispose()


# ============================================================================
# 14. Edge cases and additional coverage
# ============================================================================


class TestEdgeCases:
    """Additional edge case tests for complete coverage."""

    def test_from_env_port_zero(self, monkeypatch, clean_env):
        monkeypatch.setenv("DB_PORT", "0")
        ds = DatabaseSettings.from_env()
        assert ds.port == 0

    def test_from_env_large_port(self, monkeypatch, clean_env):
        monkeypatch.setenv("DB_PORT", "65535")
        ds = DatabaseSettings.from_env()
        assert ds.port == 65535

    def test_from_env_negative_port(self, monkeypatch, clean_env):
        monkeypatch.setenv("DB_PORT", "-1")
        ds = DatabaseSettings.from_env()
        assert ds.port == -1  # No validation in the code

    def test_sqlalchemy_url_with_ipv4_host(self):
        ds = DatabaseSettings("192.168.1.100", 5432, "db", "u", "p", {})
        url = ds.sqlalchemy_url()
        assert "192.168.1.100" in url

    def test_from_env_with_whitespace_in_host(self, monkeypatch, clean_env):
        monkeypatch.setenv("DB_HOST", "  spaced-host  ")
        ds = DatabaseSettings.from_env()
        # os.getenv does not strip whitespace
        assert ds.host == "  spaced-host  "

    def test_settings_with_unicode_password(self):
        ds = DatabaseSettings("h", 1, "db", "u", "p\u00e4ssw\u00f6rd", {})
        url = ds.sqlalchemy_url()
        assert "p\u00e4ssw\u00f6rd" in url

    @patch("db.config.create_engine")
    def test_get_engine_with_sslmode_env(
        self, mock_create_engine, monkeypatch, clean_env
    ):
        monkeypatch.setenv("DB_HOST", "prodhost")
        monkeypatch.setenv("DB_PORT", "5432")
        monkeypatch.setenv("DB_NAME", "proddb")
        monkeypatch.setenv("DB_USER", "admin")
        monkeypatch.setenv("DB_PASSWORD", "secret")
        monkeypatch.setenv("DB_SSLMODE", "verify-full")
        mock_create_engine.return_value = MagicMock()
        get_engine()
        url_arg = mock_create_engine.call_args[0][0]
        assert "prodhost" in url_arg
        assert "proddb" in url_arg
        assert "sslmode=verify-full" in url_arg

    def test_sqlalchemy_url_with_long_dbname(self):
        long_name = "a" * 200
        ds = DatabaseSettings("h", 1, long_name, "u", "p", {})
        url = ds.sqlalchemy_url()
        assert long_name in url

    def test_sqlalchemy_url_with_empty_password(self):
        ds = DatabaseSettings("h", 1, "db", "u", "", {})
        url = ds.sqlalchemy_url()
        assert ":@" in url

    def test_sqlalchemy_url_port_in_url(self):
        ds = DatabaseSettings("h", 0, "db", "u", "p", {})
        url = ds.sqlalchemy_url()
        assert ":0/" in url

    def test_from_env_partial_override(self, monkeypatch, clean_env):
        """Override only host and port, keep rest as defaults."""
        monkeypatch.setenv("DB_HOST", "custom-host")
        monkeypatch.setenv("DB_PORT", "9876")
        ds = DatabaseSettings.from_env()
        assert ds.host == "custom-host"
        assert ds.port == 9876
        assert ds.name == "vectordb"
        assert ds.user == "vectoruser"
        assert ds.password == "friday"
        assert ds.options == {}

    def test_from_env_only_password_override(self, monkeypatch, clean_env):
        monkeypatch.setenv("DB_PASSWORD", "new_pass")
        ds = DatabaseSettings.from_env()
        assert ds.password == "new_pass"
        assert ds.host == "localhost"

    def test_from_env_sslmode_with_custom_host(self, monkeypatch, clean_env):
        monkeypatch.setenv("DB_HOST", "secure.db.com")
        monkeypatch.setenv("DB_SSLMODE", "require")
        ds = DatabaseSettings.from_env()
        assert ds.host == "secure.db.com"
        assert ds.options == {"sslmode": "require"}

    def test_sqlalchemy_url_from_env_with_ssl(self, monkeypatch, clean_env):
        monkeypatch.setenv("DB_HOST", "secure.db.com")
        monkeypatch.setenv("DB_PORT", "5433")
        monkeypatch.setenv("DB_NAME", "securedb")
        monkeypatch.setenv("DB_USER", "secureuser")
        monkeypatch.setenv("DB_PASSWORD", "securepass")
        monkeypatch.setenv("DB_SSLMODE", "require")
        ds = DatabaseSettings.from_env()
        url = ds.sqlalchemy_url()
        expected = "postgresql+psycopg://secureuser:securepass@secure.db.com:5433/securedb?sslmode=require"
        assert url == expected

    @patch("db.utils.get_engine")
    def test_get_schema_snapshot_calls_get_engine_when_none(self, mock_get_engine):
        from db.utils import get_schema_snapshot

        mock_engine = create_engine("sqlite://", echo=False)
        mock_get_engine.return_value = mock_engine
        get_schema_snapshot(engine=None)
        mock_get_engine.assert_called_once()
        mock_engine.dispose()

    @patch("db.utils.get_engine")
    def test_create_all_calls_get_engine_when_none(self, mock_get_engine):
        from db.utils import create_all

        mock_engine = create_engine("sqlite://", echo=False)
        mock_get_engine.return_value = mock_engine
        create_all(engine=None)
        mock_get_engine.assert_called_once()
        mock_engine.dispose()
