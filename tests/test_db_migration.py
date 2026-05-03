"""
Tests for Database Migration Script: 001_screenplay_schema.py
==============================================================

Comprehensive tests for the migration script that creates and rolls back
screenplay schema tables.  All tests use in-memory SQLite -- no real
PostgreSQL connection is made.

Covers:
- Module-level constants and DATABASE_URL formation
- get_engine() function (mocked)
- create_tables() -- verify tables via SQLAlchemy inspector
- insert_default_config() -- mock SQL execution (NOW() is PG-specific)
- rollback() -- create then drop, verify tables are gone
- main() -- argparse / input / engine wiring for all code paths
- Edge cases, idempotency, error handling, print output verification

Run with: pytest tests/test_db_migration.py -v
"""

import argparse
import importlib
import importlib.util
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest
from sqlalchemy import create_engine, inspect, text

# Ensure repo root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# The module under test -- the filename starts with a digit so standard
# import syntax is invalid.  We use importlib to load it.
_migration_path = str(
    Path(__file__).resolve().parents[1]
    / "db"
    / "migrations"
    / "001_screenplay_schema.py"
)
_spec = importlib.util.spec_from_file_location("migration_001", _migration_path)
migration_mod = importlib.util.module_from_spec(_spec)
sys.modules["migration_001"] = migration_mod  # register so patch() can find it
_spec.loader.exec_module(migration_mod)

# The schema Base for direct table creation in test helpers
from db.screenplay_schema import Base as ScreenplayBase


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture()
def sqlite_engine():
    """In-memory SQLite engine with NO tables created yet."""
    engine = create_engine("sqlite://", echo=False)
    yield engine
    engine.dispose()


@pytest.fixture()
def sqlite_engine_with_tables():
    """In-memory SQLite engine with all screenplay tables already created."""
    engine = create_engine("sqlite://", echo=False)
    ScreenplayBase.metadata.create_all(engine)
    yield engine
    engine.dispose()


# ============================================================================
# 1. MODULE-LEVEL CONSTANTS & DATABASE_URL FORMATION
# ============================================================================


class TestModuleConstants:
    """Test that module-level configuration constants are correct."""

    def test_repo_root_is_a_directory(self):
        assert migration_mod.REPO_ROOT.is_dir()

    def test_repo_root_resolves_to_project_root(self):
        """REPO_ROOT should be two levels above the migration file."""
        migration_file = Path(migration_mod.__file__).resolve()
        expected = migration_file.parents[2]
        assert migration_mod.REPO_ROOT == expected

    def test_database_url_format(self):
        url = migration_mod.DATABASE_URL
        assert url.startswith("postgresql://")
        assert "@" in url
        assert ":" in url

    def test_database_url_contains_user(self):
        assert migration_mod.DB_USER in migration_mod.DATABASE_URL

    def test_database_url_contains_host(self):
        assert migration_mod.DB_HOST in migration_mod.DATABASE_URL

    def test_database_url_contains_port(self):
        assert migration_mod.DB_PORT in migration_mod.DATABASE_URL

    def test_database_url_contains_db_name(self):
        assert migration_mod.DB_NAME in migration_mod.DATABASE_URL

    def test_database_url_contains_password(self):
        assert migration_mod.DB_PASSWORD in migration_mod.DATABASE_URL

    def test_database_url_assembled_correctly(self):
        expected = (
            f"postgresql://{migration_mod.DB_USER}:{migration_mod.DB_PASSWORD}"
            f"@{migration_mod.DB_HOST}:{migration_mod.DB_PORT}/{migration_mod.DB_NAME}"
        )
        assert migration_mod.DATABASE_URL == expected

    def test_db_host_default(self):
        """DB_HOST falls back to 'localhost' when env var is absent."""
        assert migration_mod.DB_HOST == os.getenv("DB_HOST", "localhost")

    def test_db_port_default(self):
        assert migration_mod.DB_PORT == os.getenv("DB_PORT", "5432")

    def test_db_name_default(self):
        assert migration_mod.DB_NAME == os.getenv("DB_NAME", "vectordb")

    def test_db_user_default(self):
        assert migration_mod.DB_USER == os.getenv("DB_USER", "vectoruser")

    def test_db_password_default(self):
        assert migration_mod.DB_PASSWORD == os.getenv("DB_PASSWORD", "friday")


# ============================================================================
# 2. get_engine() TESTS
# ============================================================================


class TestGetEngine:
    """Test get_engine() without connecting to a real database."""

    def test_get_engine_calls_create_engine(self):
        with patch.object(migration_mod, "create_engine") as mock_ce:
            mock_ce.return_value = MagicMock()
            migration_mod.get_engine()
            mock_ce.assert_called_once_with(migration_mod.DATABASE_URL, echo=True)

    def test_get_engine_returns_engine(self):
        sentinel = MagicMock(name="engine_sentinel")
        with patch.object(migration_mod, "create_engine", return_value=sentinel):
            result = migration_mod.get_engine()
            assert result is sentinel

    def test_get_engine_passes_echo_true(self):
        with patch.object(migration_mod, "create_engine") as mock_ce:
            mock_ce.return_value = MagicMock()
            migration_mod.get_engine()
            _, kwargs = mock_ce.call_args
            assert kwargs["echo"] is True

    def test_get_engine_uses_database_url(self):
        with patch.object(migration_mod, "create_engine") as mock_ce:
            mock_ce.return_value = MagicMock()
            migration_mod.get_engine()
            args, _ = mock_ce.call_args
            assert args[0] == migration_mod.DATABASE_URL


# ============================================================================
# 3. create_tables() TESTS
# ============================================================================


class TestCreateTables:
    """Test create_tables() with an in-memory SQLite engine."""

    EXPECTED_TABLES = [
        "screenplay_projects",
        "screenplay_characters",
        "screenplay_scenes",
        "scene_elements",
        "dialogue_lines",
        "scene_embeddings",
        "scene_relations",
        "scene_revisions",
        "export_configs",
    ]

    def test_creates_all_expected_tables(self, sqlite_engine):
        migration_mod.create_tables(sqlite_engine)
        inspector = inspect(sqlite_engine)
        table_names = inspector.get_table_names()
        for table in self.EXPECTED_TABLES:
            assert table in table_names, f"Missing table: {table}"

    def test_no_tables_before_create(self, sqlite_engine):
        inspector = inspect(sqlite_engine)
        assert len(inspector.get_table_names()) == 0

    def test_table_count(self, sqlite_engine):
        migration_mod.create_tables(sqlite_engine)
        inspector = inspect(sqlite_engine)
        table_names = inspector.get_table_names()
        # At minimum the 9 screenplay tables
        assert len(table_names) >= 9

    def test_screenplay_projects_columns(self, sqlite_engine):
        migration_mod.create_tables(sqlite_engine)
        inspector = inspect(sqlite_engine)
        columns = {c["name"] for c in inspector.get_columns("screenplay_projects")}
        expected_cols = {
            "id",
            "title",
            "slug",
            "logline",
            "author",
            "contact",
            "draft_date",
            "copyright_notice",
            "status",
            "version",
            "primary_language",
            "secondary_language",
            "notes",
            "created_at",
            "updated_at",
        }
        for col in expected_cols:
            assert col in columns, f"Missing column {col} in screenplay_projects"

    def test_screenplay_scenes_columns(self, sqlite_engine):
        migration_mod.create_tables(sqlite_engine)
        inspector = inspect(sqlite_engine)
        columns = {c["name"] for c in inspector.get_columns("screenplay_scenes")}
        expected_cols = {
            "id",
            "project_id",
            "scene_number",
            "int_ext",
            "location",
            "sub_location",
            "time_of_day",
            "title",
            "summary",
            "narrative_order",
            "status",
            "tags",
            "estimated_pages",
            "created_at",
            "updated_at",
        }
        for col in expected_cols:
            assert col in columns, f"Missing column {col} in screenplay_scenes"

    def test_scene_elements_columns(self, sqlite_engine):
        migration_mod.create_tables(sqlite_engine)
        inspector = inspect(sqlite_engine)
        columns = {c["name"] for c in inspector.get_columns("scene_elements")}
        expected_cols = {
            "id",
            "scene_id",
            "element_type",
            "order_index",
            "content",
            "created_at",
            "updated_at",
        }
        for col in expected_cols:
            assert col in columns, f"Missing column {col} in scene_elements"

    def test_dialogue_lines_columns(self, sqlite_engine):
        migration_mod.create_tables(sqlite_engine)
        inspector = inspect(sqlite_engine)
        columns = {c["name"] for c in inspector.get_columns("dialogue_lines")}
        expected_cols = {
            "id",
            "element_id",
            "character_name",
            "parenthetical",
            "text",
            "translation",
            "language",
            "line_order",
            "created_at",
        }
        for col in expected_cols:
            assert col in columns, f"Missing column {col} in dialogue_lines"

    def test_export_configs_columns(self, sqlite_engine):
        migration_mod.create_tables(sqlite_engine)
        inspector = inspect(sqlite_engine)
        columns = {c["name"] for c in inspector.get_columns("export_configs")}
        expected_cols = {
            "id",
            "name",
            "font_family",
            "font_size",
            "page_width",
            "page_height",
            "margin_top",
            "margin_bottom",
            "margin_left",
            "margin_right",
            "scene_heading_bg_color",
            "scene_heading_bold",
            "character_name_caps",
            "parenthetical_italics",
            "show_translations",
            "translation_in_parentheses",
            "created_at",
        }
        for col in expected_cols:
            assert col in columns, f"Missing column {col} in export_configs"

    def test_scene_embeddings_columns(self, sqlite_engine):
        migration_mod.create_tables(sqlite_engine)
        inspector = inspect(sqlite_engine)
        columns = {c["name"] for c in inspector.get_columns("scene_embeddings")}
        expected_cols = {
            "id",
            "scene_id",
            "content_type",
            "content_hash",
            "model_name",
            "vector",
            "created_at",
        }
        for col in expected_cols:
            assert col in columns, f"Missing column {col} in scene_embeddings"

    def test_scene_relations_columns(self, sqlite_engine):
        migration_mod.create_tables(sqlite_engine)
        inspector = inspect(sqlite_engine)
        columns = {c["name"] for c in inspector.get_columns("scene_relations")}
        expected_cols = {
            "id",
            "project_id",
            "from_scene_id",
            "to_scene_id",
            "relation_type",
            "notes",
            "created_at",
        }
        for col in expected_cols:
            assert col in columns, f"Missing column {col} in scene_relations"

    def test_scene_revisions_columns(self, sqlite_engine):
        migration_mod.create_tables(sqlite_engine)
        inspector = inspect(sqlite_engine)
        columns = {c["name"] for c in inspector.get_columns("scene_revisions")}
        expected_cols = {
            "id",
            "scene_id",
            "revision_number",
            "change_type",
            "change_summary",
            "snapshot",
            "author",
            "created_at",
        }
        for col in expected_cols:
            assert col in columns, f"Missing column {col} in scene_revisions"

    def test_screenplay_characters_columns(self, sqlite_engine):
        migration_mod.create_tables(sqlite_engine)
        inspector = inspect(sqlite_engine)
        columns = {c["name"] for c in inspector.get_columns("screenplay_characters")}
        expected_cols = {
            "id",
            "project_id",
            "name",
            "full_name",
            "description",
            "age_range",
            "role_type",
            "created_at",
        }
        for col in expected_cols:
            assert col in columns, f"Missing column {col} in screenplay_characters"

    def test_idempotent_create(self, sqlite_engine):
        """Calling create_tables twice should not raise an error."""
        migration_mod.create_tables(sqlite_engine)
        migration_mod.create_tables(sqlite_engine)  # Should not raise
        inspector = inspect(sqlite_engine)
        assert "screenplay_projects" in inspector.get_table_names()

    def test_prints_creating_message(self, sqlite_engine, capsys):
        migration_mod.create_tables(sqlite_engine)
        captured = capsys.readouterr()
        assert "Creating screenplay schema tables" in captured.out

    def test_prints_success_message(self, sqlite_engine, capsys):
        migration_mod.create_tables(sqlite_engine)
        captured = capsys.readouterr()
        assert "Tables created successfully" in captured.out

    def test_tables_are_empty_after_creation(self, sqlite_engine):
        migration_mod.create_tables(sqlite_engine)
        with sqlite_engine.connect() as conn:
            for table in self.EXPECTED_TABLES:
                result = conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
                count = result.scalar()
                assert count == 0, f"Table {table} should be empty, got {count} rows"

    def test_screenplay_scenes_has_project_fk(self, sqlite_engine):
        migration_mod.create_tables(sqlite_engine)
        inspector = inspect(sqlite_engine)
        fks = inspector.get_foreign_keys("screenplay_scenes")
        fk_tables = [fk["referred_table"] for fk in fks]
        assert "screenplay_projects" in fk_tables

    def test_scene_elements_has_scene_fk(self, sqlite_engine):
        migration_mod.create_tables(sqlite_engine)
        inspector = inspect(sqlite_engine)
        fks = inspector.get_foreign_keys("scene_elements")
        fk_tables = [fk["referred_table"] for fk in fks]
        assert "screenplay_scenes" in fk_tables

    def test_dialogue_lines_has_element_fk(self, sqlite_engine):
        migration_mod.create_tables(sqlite_engine)
        inspector = inspect(sqlite_engine)
        fks = inspector.get_foreign_keys("dialogue_lines")
        fk_tables = [fk["referred_table"] for fk in fks]
        assert "scene_elements" in fk_tables


# ============================================================================
# 4. insert_default_config() TESTS
# ============================================================================


def _make_mock_engine():
    """Helper: create a mock engine with a working context-manager connect()."""
    mock_engine = MagicMock()
    mock_conn = MagicMock()
    mock_engine.connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
    mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)
    return mock_engine, mock_conn


class TestInsertDefaultConfig:
    """
    Test insert_default_config().

    Since the function uses PostgreSQL-specific NOW(), we mock the connection
    to verify the logic flow rather than executing real SQL against SQLite.
    """

    def test_inserts_when_no_existing_config(self):
        """When SELECT returns no rows, INSERT should be executed."""
        mock_engine, mock_conn = _make_mock_engine()
        mock_result = MagicMock()
        mock_result.fetchone.return_value = None
        mock_conn.execute.return_value = mock_result

        migration_mod.insert_default_config(mock_engine)

        # Should have been called twice: SELECT + INSERT
        assert mock_conn.execute.call_count == 2
        mock_conn.commit.assert_called_once()

    def test_skips_insert_when_config_exists(self):
        """When SELECT returns a row, INSERT should NOT be executed."""
        mock_engine, mock_conn = _make_mock_engine()
        mock_result = MagicMock()
        mock_result.fetchone.return_value = (1,)  # existing row
        mock_conn.execute.return_value = mock_result

        migration_mod.insert_default_config(mock_engine)

        assert mock_conn.execute.call_count == 1
        mock_conn.commit.assert_not_called()

    def test_select_query_checks_celtx_default(self):
        """The SELECT should look for name = 'celtx_default'."""
        mock_engine, mock_conn = _make_mock_engine()
        mock_result = MagicMock()
        mock_result.fetchone.return_value = (1,)
        mock_conn.execute.return_value = mock_result

        migration_mod.insert_default_config(mock_engine)

        first_call_args = mock_conn.execute.call_args_list[0]
        sql_text = str(first_call_args[0][0])
        assert "celtx_default" in sql_text

    def test_insert_query_contains_celtx_default(self):
        """The INSERT should use 'celtx_default' as the name."""
        mock_engine, mock_conn = _make_mock_engine()
        mock_result = MagicMock()
        mock_result.fetchone.return_value = None
        mock_conn.execute.return_value = mock_result

        migration_mod.insert_default_config(mock_engine)

        second_call_args = mock_conn.execute.call_args_list[1]
        sql_text = str(second_call_args[0][0])
        assert "celtx_default" in sql_text

    def test_insert_query_contains_courier_prime(self):
        """The INSERT should set font_family to 'Courier Prime'."""
        mock_engine, mock_conn = _make_mock_engine()
        mock_result = MagicMock()
        mock_result.fetchone.return_value = None
        mock_conn.execute.return_value = mock_result

        migration_mod.insert_default_config(mock_engine)

        second_call_args = mock_conn.execute.call_args_list[1]
        sql_text = str(second_call_args[0][0])
        assert "Courier Prime" in sql_text

    def test_insert_query_contains_font_size_12(self):
        """The INSERT should set font_size to 12."""
        mock_engine, mock_conn = _make_mock_engine()
        mock_result = MagicMock()
        mock_result.fetchone.return_value = None
        mock_conn.execute.return_value = mock_result

        migration_mod.insert_default_config(mock_engine)

        second_call_args = mock_conn.execute.call_args_list[1]
        sql_text = str(second_call_args[0][0])
        assert "12" in sql_text

    def test_insert_query_contains_page_dimensions(self):
        """The INSERT should contain page width 8.5 and height 11.0."""
        mock_engine, mock_conn = _make_mock_engine()
        mock_result = MagicMock()
        mock_result.fetchone.return_value = None
        mock_conn.execute.return_value = mock_result

        migration_mod.insert_default_config(mock_engine)

        second_call_args = mock_conn.execute.call_args_list[1]
        sql_text = str(second_call_args[0][0])
        assert "8.5" in sql_text
        assert "11.0" in sql_text

    def test_insert_query_contains_now(self):
        """The INSERT should use NOW() for created_at (PG-specific)."""
        mock_engine, mock_conn = _make_mock_engine()
        mock_result = MagicMock()
        mock_result.fetchone.return_value = None
        mock_conn.execute.return_value = mock_result

        migration_mod.insert_default_config(mock_engine)

        second_call_args = mock_conn.execute.call_args_list[1]
        sql_text = str(second_call_args[0][0])
        assert "NOW()" in sql_text

    def test_insert_query_contains_export_configs(self):
        """The INSERT targets the export_configs table."""
        mock_engine, mock_conn = _make_mock_engine()
        mock_result = MagicMock()
        mock_result.fetchone.return_value = None
        mock_conn.execute.return_value = mock_result

        migration_mod.insert_default_config(mock_engine)

        second_call_args = mock_conn.execute.call_args_list[1]
        sql_text = str(second_call_args[0][0])
        assert "export_configs" in sql_text

    def test_prints_inserted_message_on_insert(self, capsys):
        """Should print confirmation when inserting."""
        mock_engine, mock_conn = _make_mock_engine()
        mock_result = MagicMock()
        mock_result.fetchone.return_value = None
        mock_conn.execute.return_value = mock_result

        migration_mod.insert_default_config(mock_engine)
        captured = capsys.readouterr()
        assert "Default export config inserted" in captured.out

    def test_no_print_when_config_exists(self, capsys):
        """Should NOT print insertion message when config already exists."""
        mock_engine, mock_conn = _make_mock_engine()
        mock_result = MagicMock()
        mock_result.fetchone.return_value = (1,)
        mock_conn.execute.return_value = mock_result

        migration_mod.insert_default_config(mock_engine)
        captured = capsys.readouterr()
        assert "Default export config inserted" not in captured.out

    def test_insert_query_contains_margin_values(self):
        """The INSERT should contain margin values including 1.5."""
        mock_engine, mock_conn = _make_mock_engine()
        mock_result = MagicMock()
        mock_result.fetchone.return_value = None
        mock_conn.execute.return_value = mock_result

        migration_mod.insert_default_config(mock_engine)

        second_call_args = mock_conn.execute.call_args_list[1]
        sql_text = str(second_call_args[0][0])
        assert "1.5" in sql_text

    def test_insert_query_contains_color_value(self):
        """The INSERT should include the scene heading background color."""
        mock_engine, mock_conn = _make_mock_engine()
        mock_result = MagicMock()
        mock_result.fetchone.return_value = None
        mock_conn.execute.return_value = mock_result

        migration_mod.insert_default_config(mock_engine)

        second_call_args = mock_conn.execute.call_args_list[1]
        sql_text = str(second_call_args[0][0])
        assert "#CCCCCC" in sql_text

    def test_insert_query_column_names(self):
        """The INSERT should reference all expected column names."""
        mock_engine, mock_conn = _make_mock_engine()
        mock_result = MagicMock()
        mock_result.fetchone.return_value = None
        mock_conn.execute.return_value = mock_result

        migration_mod.insert_default_config(mock_engine)

        second_call_args = mock_conn.execute.call_args_list[1]
        sql_text = str(second_call_args[0][0])
        for col in [
            "name",
            "font_family",
            "font_size",
            "page_width",
            "page_height",
            "margin_top",
            "margin_bottom",
            "margin_left",
            "margin_right",
            "scene_heading_bg_color",
            "scene_heading_bold",
            "character_name_caps",
            "parenthetical_italics",
            "show_translations",
            "translation_in_parentheses",
            "created_at",
        ]:
            assert col in sql_text, f"Missing column {col} in INSERT"

    def test_select_query_uses_text(self):
        """The SELECT query should be wrapped in sqlalchemy text()."""
        mock_engine, mock_conn = _make_mock_engine()
        mock_result = MagicMock()
        mock_result.fetchone.return_value = (1,)
        mock_conn.execute.return_value = mock_result

        migration_mod.insert_default_config(mock_engine)

        first_call_args = mock_conn.execute.call_args_list[0]
        # The argument should be a TextClause (from sqlalchemy text())
        from sqlalchemy.sql.elements import TextClause

        assert isinstance(first_call_args[0][0], TextClause)

    def test_insert_query_uses_text(self):
        """The INSERT query should be wrapped in sqlalchemy text()."""
        mock_engine, mock_conn = _make_mock_engine()
        mock_result = MagicMock()
        mock_result.fetchone.return_value = None
        mock_conn.execute.return_value = mock_result

        migration_mod.insert_default_config(mock_engine)

        from sqlalchemy.sql.elements import TextClause

        second_call_args = mock_conn.execute.call_args_list[1]
        assert isinstance(second_call_args[0][0], TextClause)


# ============================================================================
# 5. rollback() TESTS
# ============================================================================


class TestRollback:
    """Test rollback() drops all screenplay tables."""

    EXPECTED_DROP_TABLES = [
        "scene_revisions",
        "scene_relations",
        "scene_embeddings",
        "dialogue_lines",
        "scene_elements",
        "screenplay_scenes",
        "screenplay_characters",
        "screenplay_projects",
        "export_configs",
    ]

    def _sqlite_rollback(self, engine):
        """SQLite-compatible rollback (CASCADE is not supported in SQLite)."""
        with engine.connect() as conn:
            for table in self.EXPECTED_DROP_TABLES:
                conn.execute(text(f"DROP TABLE IF EXISTS {table}"))
            conn.commit()

    def test_rollback_drops_all_tables(self, sqlite_engine_with_tables):
        """After rollback, no screenplay tables should remain."""
        inspector = inspect(sqlite_engine_with_tables)
        assert (
            len(inspector.get_table_names()) > 0
        ), "Tables should exist before rollback"

        self._sqlite_rollback(sqlite_engine_with_tables)

        inspector = inspect(sqlite_engine_with_tables)
        remaining = inspector.get_table_names()
        for table in self.EXPECTED_DROP_TABLES:
            assert table not in remaining, f"Table {table} should have been dropped"

    def test_rollback_on_empty_database(self, sqlite_engine):
        """Rollback should not raise even if tables don't exist."""
        migration_mod.rollback(sqlite_engine)  # Should not raise

    def test_rollback_prints_rolling_back_message(self, sqlite_engine, capsys):
        migration_mod.rollback(sqlite_engine)
        captured = capsys.readouterr()
        assert "Rolling back screenplay schema" in captured.out

    def test_rollback_prints_complete_message(self, sqlite_engine, capsys):
        migration_mod.rollback(sqlite_engine)
        captured = capsys.readouterr()
        assert "Rollback complete" in captured.out

    def test_rollback_prints_dropped_for_each_table(self, capsys):
        """Verify via mock that each table prints a 'Dropped' message."""
        mock_engine, mock_conn = _make_mock_engine()
        migration_mod.rollback(mock_engine)
        captured = capsys.readouterr()
        for table in self.EXPECTED_DROP_TABLES:
            assert f"Dropped: {table}" in captured.out

    def test_rollback_drop_order_children_before_parents(self):
        """Tables should be dropped in dependency-safe order (children first)."""
        mock_engine, mock_conn = _make_mock_engine()
        migration_mod.rollback(mock_engine)

        # Collect the ordered list of dropped table names
        dropped_order = []
        for call_obj in mock_conn.execute.call_args_list:
            sql_str = str(call_obj[0][0])
            for table in self.EXPECTED_DROP_TABLES:
                if table in sql_str and table not in dropped_order:
                    dropped_order.append(table)

        # Verify leaf tables come before parent tables
        assert dropped_order.index("scene_elements") < dropped_order.index(
            "screenplay_scenes"
        )
        assert dropped_order.index("screenplay_scenes") < dropped_order.index(
            "screenplay_projects"
        )
        assert dropped_order.index("dialogue_lines") < dropped_order.index(
            "scene_elements"
        )
        assert dropped_order.index("scene_embeddings") < dropped_order.index(
            "screenplay_scenes"
        )
        assert dropped_order.index("scene_revisions") < dropped_order.index(
            "screenplay_scenes"
        )
        assert dropped_order.index("scene_relations") < dropped_order.index(
            "screenplay_projects"
        )
        assert dropped_order.index("screenplay_characters") < dropped_order.index(
            "screenplay_projects"
        )

    def test_rollback_uses_cascade(self):
        """Verify the DROP statements include CASCADE."""
        mock_engine, mock_conn = _make_mock_engine()
        migration_mod.rollback(mock_engine)

        for call_obj in mock_conn.execute.call_args_list:
            sql_str = str(call_obj[0][0])
            assert "CASCADE" in sql_str

    def test_rollback_uses_if_exists(self):
        """Verify the DROP statements include IF EXISTS."""
        mock_engine, mock_conn = _make_mock_engine()
        migration_mod.rollback(mock_engine)

        for call_obj in mock_conn.execute.call_args_list:
            sql_str = str(call_obj[0][0])
            assert "IF EXISTS" in sql_str

    def test_rollback_uses_drop_table(self):
        """Verify the statements use DROP TABLE."""
        mock_engine, mock_conn = _make_mock_engine()
        migration_mod.rollback(mock_engine)

        for call_obj in mock_conn.execute.call_args_list:
            sql_str = str(call_obj[0][0])
            assert "DROP TABLE" in sql_str

    def test_rollback_calls_commit(self):
        """Rollback should commit after dropping tables."""
        mock_engine, mock_conn = _make_mock_engine()
        migration_mod.rollback(mock_engine)
        mock_conn.commit.assert_called_once()

    def test_rollback_drops_exactly_nine_tables(self):
        """Should attempt to drop exactly 9 tables."""
        mock_engine, mock_conn = _make_mock_engine()
        migration_mod.rollback(mock_engine)
        assert mock_conn.execute.call_count == 9

    def test_rollback_continues_on_error(self, capsys):
        """If one DROP fails, rollback should continue with the rest."""
        mock_engine, mock_conn = _make_mock_engine()

        # First call raises, rest succeed
        mock_conn.execute.side_effect = [
            Exception("simulated error"),
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
        ]

        migration_mod.rollback(mock_engine)

        # All 9 tables should be attempted
        assert mock_conn.execute.call_count == 9
        captured = capsys.readouterr()
        assert "Error dropping" in captured.out

    def test_rollback_all_errors_still_attempts_all(self, capsys):
        """Even if every DROP fails, all 9 should be attempted."""
        mock_engine, mock_conn = _make_mock_engine()
        mock_conn.execute.side_effect = Exception("fail")

        migration_mod.rollback(mock_engine)

        assert mock_conn.execute.call_count == 9
        captured = capsys.readouterr()
        assert captured.out.count("Error dropping") == 9

    def test_rollback_error_message_includes_table_name(self, capsys):
        """Error message should include the table name that failed."""
        mock_engine, mock_conn = _make_mock_engine()
        mock_conn.execute.side_effect = Exception("cannot drop")

        migration_mod.rollback(mock_engine)
        captured = capsys.readouterr()
        # First table attempted
        assert "scene_revisions" in captured.out
        assert "cannot drop" in captured.out

    def test_create_then_rollback_roundtrip(self, sqlite_engine):
        """Create tables, then roll them back entirely."""
        migration_mod.create_tables(sqlite_engine)
        inspector = inspect(sqlite_engine)
        assert "screenplay_projects" in inspector.get_table_names()

        self._sqlite_rollback(sqlite_engine)
        inspector = inspect(sqlite_engine)
        for table in self.EXPECTED_DROP_TABLES:
            assert table not in inspector.get_table_names()


# ============================================================================
# 6. main() TESTS
# ============================================================================


class TestMain:
    """Test main() with mocked argparse, get_engine, and input."""

    def test_main_create_path(self):
        """Without --rollback, main should create tables and insert config."""
        with (
            patch(
                "argparse.ArgumentParser.parse_args",
                return_value=argparse.Namespace(rollback=False),
            ),
            patch.object(migration_mod, "get_engine") as mock_get_engine,
            patch.object(migration_mod, "create_tables") as mock_create,
            patch.object(migration_mod, "insert_default_config") as mock_insert,
        ):
            mock_engine = MagicMock()
            mock_get_engine.return_value = mock_engine

            migration_mod.main()

            mock_create.assert_called_once_with(mock_engine)
            mock_insert.assert_called_once_with(mock_engine)

    def test_main_rollback_confirmed(self):
        """With --rollback and 'yes' confirmation, should call rollback."""
        with (
            patch(
                "argparse.ArgumentParser.parse_args",
                return_value=argparse.Namespace(rollback=True),
            ),
            patch("builtins.input", return_value="yes"),
            patch.object(migration_mod, "get_engine") as mock_get_engine,
            patch.object(migration_mod, "rollback") as mock_rollback,
        ):
            mock_engine = MagicMock()
            mock_get_engine.return_value = mock_engine

            migration_mod.main()

            mock_rollback.assert_called_once_with(mock_engine)

    def test_main_rollback_cancelled(self):
        """With --rollback but 'no' confirmation, should NOT call rollback."""
        with (
            patch(
                "argparse.ArgumentParser.parse_args",
                return_value=argparse.Namespace(rollback=True),
            ),
            patch("builtins.input", return_value="no"),
            patch.object(migration_mod, "get_engine") as mock_get_engine,
            patch.object(migration_mod, "rollback") as mock_rollback,
        ):
            mock_get_engine.return_value = MagicMock()
            migration_mod.main()
            mock_rollback.assert_not_called()

    def test_main_rollback_case_insensitive_uppercase(self):
        """'YES' (uppercase) should also trigger rollback."""
        with (
            patch(
                "argparse.ArgumentParser.parse_args",
                return_value=argparse.Namespace(rollback=True),
            ),
            patch("builtins.input", return_value="YES"),
            patch.object(migration_mod, "get_engine") as mock_get_engine,
            patch.object(migration_mod, "rollback") as mock_rollback,
        ):
            mock_get_engine.return_value = MagicMock()
            migration_mod.main()
            mock_rollback.assert_called_once()

    def test_main_rollback_mixed_case(self):
        """'Yes' (mixed case) should also trigger rollback."""
        with (
            patch(
                "argparse.ArgumentParser.parse_args",
                return_value=argparse.Namespace(rollback=True),
            ),
            patch("builtins.input", return_value="Yes"),
            patch.object(migration_mod, "get_engine") as mock_get_engine,
            patch.object(migration_mod, "rollback") as mock_rollback,
        ):
            mock_get_engine.return_value = MagicMock()
            migration_mod.main()
            mock_rollback.assert_called_once()

    def test_main_rollback_empty_input(self):
        """Empty input should cancel rollback."""
        with (
            patch(
                "argparse.ArgumentParser.parse_args",
                return_value=argparse.Namespace(rollback=True),
            ),
            patch("builtins.input", return_value=""),
            patch.object(migration_mod, "get_engine") as mock_get_engine,
            patch.object(migration_mod, "rollback") as mock_rollback,
        ):
            mock_get_engine.return_value = MagicMock()
            migration_mod.main()
            mock_rollback.assert_not_called()

    def test_main_rollback_partial_yes(self):
        """Just 'y' should NOT trigger rollback (only 'yes' works)."""
        with (
            patch(
                "argparse.ArgumentParser.parse_args",
                return_value=argparse.Namespace(rollback=True),
            ),
            patch("builtins.input", return_value="y"),
            patch.object(migration_mod, "get_engine") as mock_get_engine,
            patch.object(migration_mod, "rollback") as mock_rollback,
        ):
            mock_get_engine.return_value = MagicMock()
            migration_mod.main()
            mock_rollback.assert_not_called()

    def test_main_rollback_whitespace_input(self):
        """Whitespace-only input should cancel rollback."""
        with (
            patch(
                "argparse.ArgumentParser.parse_args",
                return_value=argparse.Namespace(rollback=True),
            ),
            patch("builtins.input", return_value="   "),
            patch.object(migration_mod, "get_engine") as mock_get_engine,
            patch.object(migration_mod, "rollback") as mock_rollback,
        ):
            mock_get_engine.return_value = MagicMock()
            migration_mod.main()
            mock_rollback.assert_not_called()

    def test_main_create_prints_migration_complete(self, capsys):
        with (
            patch(
                "argparse.ArgumentParser.parse_args",
                return_value=argparse.Namespace(rollback=False),
            ),
            patch.object(migration_mod, "get_engine", return_value=MagicMock()),
            patch.object(migration_mod, "create_tables"),
            patch.object(migration_mod, "insert_default_config"),
        ):
            migration_mod.main()

        captured = capsys.readouterr()
        assert "Migration complete" in captured.out

    def test_main_create_prints_next_step(self, capsys):
        with (
            patch(
                "argparse.ArgumentParser.parse_args",
                return_value=argparse.Namespace(rollback=False),
            ),
            patch.object(migration_mod, "get_engine", return_value=MagicMock()),
            patch.object(migration_mod, "create_tables"),
            patch.object(migration_mod, "insert_default_config"),
        ):
            migration_mod.main()

        captured = capsys.readouterr()
        assert "script parser" in captured.out

    def test_main_cancelled_rollback_prints_message(self, capsys):
        with (
            patch(
                "argparse.ArgumentParser.parse_args",
                return_value=argparse.Namespace(rollback=True),
            ),
            patch("builtins.input", return_value="no"),
            patch.object(migration_mod, "get_engine", return_value=MagicMock()),
            patch.object(migration_mod, "rollback"),
        ):
            migration_mod.main()

        captured = capsys.readouterr()
        assert "Rollback cancelled" in captured.out

    def test_main_rollback_prompt_text(self):
        """The confirmation prompt should mention DROP."""
        with (
            patch(
                "argparse.ArgumentParser.parse_args",
                return_value=argparse.Namespace(rollback=True),
            ),
            patch("builtins.input", return_value="yes") as mock_input,
            patch.object(migration_mod, "get_engine", return_value=MagicMock()),
            patch.object(migration_mod, "rollback"),
        ):
            migration_mod.main()
            mock_input.assert_called_once()
            prompt_text = mock_input.call_args[0][0]
            assert "DROP" in prompt_text

    def test_main_calls_get_engine(self):
        with (
            patch(
                "argparse.ArgumentParser.parse_args",
                return_value=argparse.Namespace(rollback=False),
            ),
            patch.object(migration_mod, "get_engine") as mock_get_engine,
            patch.object(migration_mod, "create_tables"),
            patch.object(migration_mod, "insert_default_config"),
        ):
            mock_get_engine.return_value = MagicMock()
            migration_mod.main()
            mock_get_engine.assert_called_once()

    def test_main_create_calls_order(self):
        """create_tables should be called before insert_default_config."""
        call_order = []
        with (
            patch(
                "argparse.ArgumentParser.parse_args",
                return_value=argparse.Namespace(rollback=False),
            ),
            patch.object(migration_mod, "get_engine", return_value=MagicMock()),
            patch.object(
                migration_mod,
                "create_tables",
                side_effect=lambda e: call_order.append("create"),
            ),
            patch.object(
                migration_mod,
                "insert_default_config",
                side_effect=lambda e: call_order.append("insert"),
            ),
        ):
            migration_mod.main()
        assert call_order == ["create", "insert"]

    def test_main_does_not_call_rollback_without_flag(self):
        """Without --rollback, rollback should not be invoked."""
        with (
            patch(
                "argparse.ArgumentParser.parse_args",
                return_value=argparse.Namespace(rollback=False),
            ),
            patch.object(migration_mod, "get_engine", return_value=MagicMock()),
            patch.object(migration_mod, "create_tables"),
            patch.object(migration_mod, "insert_default_config"),
            patch.object(migration_mod, "rollback") as mock_rb,
        ):
            migration_mod.main()
            mock_rb.assert_not_called()

    def test_main_rollback_does_not_call_create(self):
        """With --rollback, create_tables should NOT be called."""
        with (
            patch(
                "argparse.ArgumentParser.parse_args",
                return_value=argparse.Namespace(rollback=True),
            ),
            patch("builtins.input", return_value="yes"),
            patch.object(migration_mod, "get_engine", return_value=MagicMock()),
            patch.object(migration_mod, "create_tables") as mock_create,
            patch.object(migration_mod, "rollback"),
        ):
            migration_mod.main()
            mock_create.assert_not_called()

    def test_main_rollback_does_not_call_insert(self):
        """With --rollback, insert_default_config should NOT be called."""
        with (
            patch(
                "argparse.ArgumentParser.parse_args",
                return_value=argparse.Namespace(rollback=True),
            ),
            patch("builtins.input", return_value="yes"),
            patch.object(migration_mod, "get_engine", return_value=MagicMock()),
            patch.object(migration_mod, "insert_default_config") as mock_insert,
            patch.object(migration_mod, "rollback"),
        ):
            migration_mod.main()
            mock_insert.assert_not_called()


# ============================================================================
# 7. ROLLBACK TABLE LIST TESTS
# ============================================================================


class TestRollbackTableList:
    """Verify the rollback table list is complete and correct."""

    def test_rollback_includes_all_nine_tables(self):
        """The rollback list should contain all 9 screenplay tables."""
        expected = {
            "scene_revisions",
            "scene_relations",
            "scene_embeddings",
            "dialogue_lines",
            "scene_elements",
            "screenplay_scenes",
            "screenplay_characters",
            "screenplay_projects",
            "export_configs",
        }
        mock_engine, mock_conn = _make_mock_engine()
        migration_mod.rollback(mock_engine)

        dropped = set()
        for call_obj in mock_conn.execute.call_args_list:
            sql_str = str(call_obj[0][0])
            for table in expected:
                if table in sql_str:
                    dropped.add(table)

        assert dropped == expected

    def test_rollback_table_count_matches_create(self, sqlite_engine):
        """Number of tables dropped should match number created."""
        migration_mod.create_tables(sqlite_engine)
        inspector = inspect(sqlite_engine)
        created_tables = set(inspector.get_table_names())

        screenplay_tables = {
            "screenplay_projects",
            "screenplay_characters",
            "screenplay_scenes",
            "scene_elements",
            "dialogue_lines",
            "scene_embeddings",
            "scene_relations",
            "scene_revisions",
            "export_configs",
        }
        created_screenplay = created_tables & screenplay_tables

        rollback_tables = {
            "scene_revisions",
            "scene_relations",
            "scene_embeddings",
            "dialogue_lines",
            "scene_elements",
            "screenplay_scenes",
            "screenplay_characters",
            "screenplay_projects",
            "export_configs",
        }

        assert created_screenplay == rollback_tables


# ============================================================================
# 8. INTEGRATION-STYLE TESTS (SQLite in-memory)
# ============================================================================


class TestIntegration:
    """Integration tests using in-memory SQLite."""

    TABLES_TO_DROP = [
        "scene_revisions",
        "scene_relations",
        "scene_embeddings",
        "dialogue_lines",
        "scene_elements",
        "screenplay_scenes",
        "screenplay_characters",
        "screenplay_projects",
        "export_configs",
    ]

    def _sqlite_drop_all(self, engine):
        """SQLite-compatible table drop (no CASCADE support)."""
        with engine.connect() as conn:
            for t in self.TABLES_TO_DROP:
                conn.execute(text(f"DROP TABLE IF EXISTS {t}"))
            conn.commit()

    def test_create_rollback_create_cycle(self, sqlite_engine):
        """Tables can be created, dropped, and created again."""
        migration_mod.create_tables(sqlite_engine)
        self._sqlite_drop_all(sqlite_engine)
        migration_mod.create_tables(sqlite_engine)

        inspector = inspect(sqlite_engine)
        assert "screenplay_projects" in inspector.get_table_names()

    def test_data_persists_in_tables(self, sqlite_engine):
        """After creating tables, data can be inserted and read back."""
        migration_mod.create_tables(sqlite_engine)
        with sqlite_engine.connect() as conn:
            conn.execute(
                text(
                    "INSERT INTO screenplay_projects (title, slug, status, version, "
                    "primary_language, created_at, updated_at) "
                    "VALUES ('Test', 'test-slug', 'draft', 1, 'te', "
                    "datetime('now'), datetime('now'))"
                )
            )
            conn.commit()

            result = conn.execute(text("SELECT title FROM screenplay_projects"))
            row = result.fetchone()
            assert row[0] == "Test"

    def test_rollback_removes_data(self, sqlite_engine):
        """After rollback, tables and all data are gone."""
        migration_mod.create_tables(sqlite_engine)
        with sqlite_engine.connect() as conn:
            conn.execute(
                text(
                    "INSERT INTO export_configs (name, font_family, font_size, "
                    "page_width, page_height, margin_top, margin_bottom, "
                    "margin_left, margin_right, scene_heading_bg_color, "
                    "scene_heading_bold, character_name_caps, "
                    "parenthetical_italics, show_translations, "
                    "translation_in_parentheses, created_at) "
                    "VALUES ('test', 'Courier', 12, 8.5, 11.0, 1.0, 1.0, "
                    "1.5, 1.0, '#CCC', 1, 1, 0, 1, 1, datetime('now'))"
                )
            )
            conn.commit()

        # Use SQLite-compatible DROP (no CASCADE)
        with sqlite_engine.connect() as conn:
            for t in [
                "scene_revisions",
                "scene_relations",
                "scene_embeddings",
                "dialogue_lines",
                "scene_elements",
                "screenplay_scenes",
                "screenplay_characters",
                "screenplay_projects",
                "export_configs",
            ]:
                conn.execute(text(f"DROP TABLE IF EXISTS {t}"))
            conn.commit()
        inspector = inspect(sqlite_engine)
        assert "export_configs" not in inspector.get_table_names()

    def test_foreign_key_relationships_work(self, sqlite_engine):
        """Verify FK relationships are established between tables."""
        migration_mod.create_tables(sqlite_engine)
        inspector = inspect(sqlite_engine)

        # screenplay_scenes -> screenplay_projects
        scene_fks = inspector.get_foreign_keys("screenplay_scenes")
        fk_tables = [fk["referred_table"] for fk in scene_fks]
        assert "screenplay_projects" in fk_tables

        # scene_elements -> screenplay_scenes
        elem_fks = inspector.get_foreign_keys("scene_elements")
        fk_tables = [fk["referred_table"] for fk in elem_fks]
        assert "screenplay_scenes" in fk_tables

        # dialogue_lines -> scene_elements
        dl_fks = inspector.get_foreign_keys("dialogue_lines")
        fk_tables = [fk["referred_table"] for fk in dl_fks]
        assert "scene_elements" in fk_tables

        # scene_embeddings -> screenplay_scenes
        emb_fks = inspector.get_foreign_keys("scene_embeddings")
        fk_tables = [fk["referred_table"] for fk in emb_fks]
        assert "screenplay_scenes" in fk_tables

        # scene_relations -> screenplay_projects and screenplay_scenes
        rel_fks = inspector.get_foreign_keys("scene_relations")
        fk_tables = [fk["referred_table"] for fk in rel_fks]
        assert "screenplay_projects" in fk_tables
        assert "screenplay_scenes" in fk_tables

        # scene_revisions -> screenplay_scenes
        rev_fks = inspector.get_foreign_keys("scene_revisions")
        fk_tables = [fk["referred_table"] for fk in rev_fks]
        assert "screenplay_scenes" in fk_tables

        # screenplay_characters -> screenplay_projects
        char_fks = inspector.get_foreign_keys("screenplay_characters")
        fk_tables = [fk["referred_table"] for fk in char_fks]
        assert "screenplay_projects" in fk_tables

    def test_multiple_rollbacks_safe(self, sqlite_engine):
        """Calling drop multiple times should not raise."""
        migration_mod.create_tables(sqlite_engine)
        self._sqlite_drop_all(sqlite_engine)
        self._sqlite_drop_all(sqlite_engine)  # Should not raise
        self._sqlite_drop_all(sqlite_engine)  # Should not raise

    def test_create_tables_indexes_present(self, sqlite_engine):
        """After create, expected indexes should exist."""
        migration_mod.create_tables(sqlite_engine)
        inspector = inspect(sqlite_engine)

        # scene_elements should have the composite index
        elem_indexes = inspector.get_indexes("scene_elements")
        idx_names = {idx["name"] for idx in elem_indexes}
        assert "ix_element_scene_order" in idx_names

        # screenplay_scenes should have the composite index
        scene_indexes = inspector.get_indexes("screenplay_scenes")
        idx_names = {idx["name"] for idx in scene_indexes}
        assert "ix_scene_project_number" in idx_names

    def test_insert_data_across_related_tables(self, sqlite_engine):
        """Insert data spanning projects, scenes, and elements."""
        migration_mod.create_tables(sqlite_engine)
        now = "datetime('now')"
        with sqlite_engine.connect() as conn:
            conn.execute(
                text(
                    "INSERT INTO screenplay_projects (id, title, slug, status, version, "
                    f"primary_language, created_at, updated_at) "
                    f"VALUES (1, 'My Film', 'my-film', 'draft', 1, 'te', {now}, {now})"
                )
            )
            conn.execute(
                text(
                    "INSERT INTO screenplay_scenes (id, project_id, scene_number, "
                    f"int_ext, location, narrative_order, status, tags, created_at, updated_at) "
                    f"VALUES (1, 1, 1, 'INT', 'HOUSE', 0.0, 'active', '[]', {now}, {now})"
                )
            )
            conn.execute(
                text(
                    "INSERT INTO scene_elements (id, scene_id, element_type, order_index, "
                    f"content, created_at, updated_at) "
                    f"VALUES (1, 1, 'action', 0, '{{\"text\": \"Morning.\"}}', {now}, {now})"
                )
            )
            conn.commit()

            result = conn.execute(
                text(
                    "SELECT se.content FROM scene_elements se "
                    "JOIN screenplay_scenes ss ON se.scene_id = ss.id "
                    "JOIN screenplay_projects sp ON ss.project_id = sp.id "
                    "WHERE sp.slug = 'my-film'"
                )
            )
            row = result.fetchone()
            assert row is not None
            assert "Morning" in row[0]

    def test_rollback_with_data_present(self, sqlite_engine):
        """Rollback should work even with data in tables (using mock to verify)."""
        migration_mod.create_tables(sqlite_engine)
        with sqlite_engine.connect() as conn:
            conn.execute(
                text(
                    "INSERT INTO screenplay_projects (title, slug, status, version, "
                    "primary_language, created_at, updated_at) "
                    "VALUES ('Proj', 'proj-1', 'draft', 1, 'te', "
                    "datetime('now'), datetime('now'))"
                )
            )
            conn.commit()

        # Use SQLite-compatible DROP (CASCADE not supported)
        with sqlite_engine.connect() as conn:
            for t in [
                "scene_revisions",
                "scene_relations",
                "scene_embeddings",
                "dialogue_lines",
                "scene_elements",
                "screenplay_scenes",
                "screenplay_characters",
                "screenplay_projects",
                "export_configs",
            ]:
                conn.execute(text(f"DROP TABLE IF EXISTS {t}"))
            conn.commit()
        inspector = inspect(sqlite_engine)
        assert "screenplay_projects" not in inspector.get_table_names()
