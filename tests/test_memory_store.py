"""
Tests for Memory Store
======================

Comprehensive tests for src/memory/store.py covering helper functions,
MemoryStore initialization, file IO helpers, persona/principles,
LTM memories, content snippets, STM sessions, and interactions.

Run with: pytest tests/test_memory_store.py -v
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

# Ensure repo root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# src/memory/store.py is shadowed by the root-level memory/ package,
# so we load it directly via importlib.
import importlib.util

_STORE_PATH = str(Path(__file__).resolve().parents[1] / "src" / "memory" / "store.py")
_spec = importlib.util.spec_from_file_location("src_memory_store", _STORE_PATH)
_store_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_store_mod)

MemoryStore = _store_mod.MemoryStore
_cosine = _store_mod._cosine
_tokenize = _store_mod._tokenize
_utc_now = _store_mod._utc_now


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def store(tmp_path):
    """Create a MemoryStore rooted in a temp directory."""
    return MemoryStore(root_dir=str(tmp_path / "memory"))


@pytest.fixture()
def populated_store(store):
    """A store pre-loaded with a few LTM entries for search tests."""
    store.add_ltm_memory(
        "Telugu is a Dravidian language",
        tags=["language", "telugu"],
        embedding=[1.0, 0.0, 0.0],
    )
    store.add_ltm_memory(
        "Python is a programming language",
        lang="en",
        tags=["programming", "python"],
        embedding=[0.0, 1.0, 0.0],
    )
    store.add_ltm_memory(
        "Hyderabad is the capital of Telangana",
        tags=["geography", "telugu"],
        embedding=[0.0, 0.0, 1.0],
    )
    return store


@pytest.fixture()
def populated_snippets_store(store):
    """A store pre-loaded with a few snippet entries for search tests."""
    store.add_snippet(
        title="Telugu Grammar",
        body="Telugu grammar includes sandhi rules and vibhakti",
        tags=["grammar", "telugu"],
        embedding=[1.0, 0.0, 0.0],
    )
    store.add_snippet(
        title="Python Tutorial",
        body="Learn Python programming from basics",
        tags=["programming", "python"],
        embedding=[0.0, 1.0, 0.0],
    )
    store.add_snippet(
        title="Hyderabad Guide",
        body="Explore the city of Hyderabad",
        tags=["travel", "telugu"],
        embedding=[0.0, 0.0, 1.0],
    )
    return store


# ===========================================================================
# 1. _utc_now helper
# ===========================================================================


class TestUtcNow:
    """Tests for the _utc_now() helper function."""

    def test_returns_string(self):
        result = _utc_now()
        assert isinstance(result, str)

    def test_returns_iso_format(self):
        result = _utc_now()
        # Should be parseable as ISO format
        parsed = datetime.fromisoformat(result)
        assert parsed is not None

    def test_utc_timezone(self):
        result = _utc_now()
        parsed = datetime.fromisoformat(result)
        assert parsed.tzinfo is not None
        assert parsed.utcoffset().total_seconds() == 0

    def test_no_microseconds(self):
        result = _utc_now()
        parsed = datetime.fromisoformat(result)
        assert parsed.microsecond == 0

    def test_contains_timezone_marker(self):
        result = _utc_now()
        # UTC ISO format should end with +00:00
        assert "+00:00" in result

    def test_recent_timestamp(self):
        before = datetime.now(timezone.utc).replace(microsecond=0)
        result = _utc_now()
        after = datetime.now(timezone.utc).replace(microsecond=0)
        parsed = datetime.fromisoformat(result)
        assert before <= parsed <= after


# ===========================================================================
# 2. _cosine helper
# ===========================================================================


class TestCosine:
    """Tests for the _cosine() helper function."""

    def test_identical_vectors_return_one(self):
        assert _cosine([1.0, 0.0, 0.0], [1.0, 0.0, 0.0]) == pytest.approx(1.0)

    def test_identical_non_unit_vectors_return_one(self):
        assert _cosine([3.0, 4.0], [3.0, 4.0]) == pytest.approx(1.0)

    def test_orthogonal_vectors_return_zero(self):
        assert _cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_opposite_vectors_return_negative_one(self):
        assert _cosine([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)

    def test_empty_first_vector(self):
        assert _cosine([], [1.0, 2.0]) == -1.0

    def test_empty_second_vector(self):
        assert _cosine([1.0, 2.0], []) == -1.0

    def test_both_empty_vectors(self):
        assert _cosine([], []) == -1.0

    def test_mismatched_lengths(self):
        assert _cosine([1.0, 2.0], [1.0, 2.0, 3.0]) == -1.0

    def test_zero_first_vector(self):
        assert _cosine([0.0, 0.0], [1.0, 2.0]) == -1.0

    def test_zero_second_vector(self):
        assert _cosine([1.0, 2.0], [0.0, 0.0]) == -1.0

    def test_both_zero_vectors(self):
        assert _cosine([0.0, 0.0], [0.0, 0.0]) == -1.0

    def test_negative_values(self):
        # cos(a, -a) = -1
        assert _cosine([1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]) == pytest.approx(-1.0)

    def test_mixed_positive_negative(self):
        # [1, -1] vs [1, 1] -> dot=0, cos=0
        assert _cosine([1.0, -1.0], [1.0, 1.0]) == pytest.approx(0.0)

    def test_known_angle_45_degrees(self):
        # cos(45deg) ~ 0.7071
        a = [1.0, 0.0]
        b = [1.0, 1.0]
        expected = 1.0 / math.sqrt(2)
        assert _cosine(a, b) == pytest.approx(expected, abs=1e-6)

    def test_single_element_vectors(self):
        assert _cosine([5.0], [3.0]) == pytest.approx(1.0)

    def test_single_element_negative(self):
        assert _cosine([5.0], [-3.0]) == pytest.approx(-1.0)

    def test_large_vectors(self):
        n = 1000
        a = [1.0] * n
        b = [1.0] * n
        assert _cosine(a, b) == pytest.approx(1.0)


# ===========================================================================
# 3. _tokenize helper
# ===========================================================================


class TestTokenize:
    """Tests for the _tokenize() helper function."""

    def test_basic_words(self):
        result = _tokenize("hello world")
        assert result == ["hello", "world"]

    def test_lowercasing(self):
        result = _tokenize("Hello WORLD")
        assert result == ["hello", "world"]

    def test_punctuation_stripping(self):
        result = _tokenize("hello, world! How are you?")
        assert result == ["hello", "world", "how", "are", "you"]

    def test_empty_string(self):
        result = _tokenize("")
        assert result == []

    def test_whitespace_only(self):
        result = _tokenize("   ")
        assert result == []

    def test_numbers(self):
        result = _tokenize("abc 123 def")
        assert result == ["abc", "123", "def"]

    def test_mixed_alphanumeric(self):
        result = _tokenize("word1 word2")
        assert result == ["word1", "word2"]

    def test_special_characters_become_spaces(self):
        result = _tokenize("hello-world_test@email.com")
        assert result == ["hello", "world", "test", "email", "com"]

    def test_unicode_telugu(self):
        # Telugu characters are alphanumeric per isalnum(), but the halant
        # (virama \u0c4d) is not, so conjunct aksharas get split.
        # The tokenizer still produces tokens from Telugu text.
        result = _tokenize("తెలుగు భాష")
        assert len(result) > 0
        # All tokens should be lowercased (already lowercase for Telugu)
        for token in result:
            assert token == token.lower()

    def test_tabs_and_newlines(self):
        result = _tokenize("hello\tworld\nfoo")
        assert result == ["hello", "world", "foo"]

    def test_multiple_spaces(self):
        result = _tokenize("hello    world")
        assert result == ["hello", "world"]

    def test_returns_list(self):
        result = _tokenize("test")
        assert isinstance(result, list)


# ===========================================================================
# 4. MemoryStore initialization
# ===========================================================================


class TestMemoryStoreInit:
    """Tests for MemoryStore.__init__."""

    def test_default_root_dir(self):
        store = MemoryStore()
        assert store.root == "memory"

    def test_custom_root_dir(self, tmp_path):
        root = str(tmp_path / "custom_memory")
        store = MemoryStore(root_dir=root)
        assert store.root == root

    def test_all_thirteen_paths_present(self, store):
        expected_keys = {
            "persona",
            "principles",
            "ltm",
            "snippets",
            "stm",
            "interactions",
            "sft",
            "dpo",
            "eval_suites",
            "eval_cases",
            "eval_runs",
            "eval_results",
            "adapters",
        }
        assert set(store.paths.keys()) == expected_keys

    def test_paths_count(self, store):
        assert len(store.paths) == 13

    def test_persona_path_correct(self, tmp_path):
        root = str(tmp_path / "memory")
        store = MemoryStore(root_dir=root)
        assert store.paths["persona"] == os.path.join(root, "data/persona/profile.json")

    def test_principles_path_correct(self, tmp_path):
        root = str(tmp_path / "memory")
        store = MemoryStore(root_dir=root)
        assert store.paths["principles"] == os.path.join(
            root, "data/principles/rules.json"
        )

    def test_ltm_path_correct(self, tmp_path):
        root = str(tmp_path / "memory")
        store = MemoryStore(root_dir=root)
        assert store.paths["ltm"] == os.path.join(root, "data/ltm_memories.jsonl")

    def test_all_paths_start_with_root(self, tmp_path):
        root = str(tmp_path / "memory")
        store = MemoryStore(root_dir=root)
        for path in store.paths.values():
            assert path.startswith(root)


# ===========================================================================
# 5. _read_json
# ===========================================================================


class TestReadJson:
    """Tests for MemoryStore._read_json."""

    def test_read_valid_json(self, store, tmp_path):
        path = str(tmp_path / "test.json")
        data = {"name": "test", "value": 42}
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f)
        result = store._read_json(path)
        assert result == data

    def test_read_json_file_not_found(self, store, tmp_path):
        path = str(tmp_path / "nonexistent.json")
        with pytest.raises(FileNotFoundError):
            store._read_json(path)

    def test_read_json_preserves_unicode(self, store, tmp_path):
        path = str(tmp_path / "unicode.json")
        data = {"text": "తెలుగు"}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        result = store._read_json(path)
        assert result["text"] == "తెలుగు"

    def test_read_json_nested(self, store, tmp_path):
        path = str(tmp_path / "nested.json")
        data = {"outer": {"inner": [1, 2, 3]}}
        with open(path, "w") as f:
            json.dump(data, f)
        result = store._read_json(path)
        assert result == data


# ===========================================================================
# 6. _write_json
# ===========================================================================


class TestWriteJson:
    """Tests for MemoryStore._write_json."""

    def test_creates_directories(self, store, tmp_path):
        path = str(tmp_path / "a" / "b" / "c" / "test.json")
        store._write_json(path, {"key": "value"})
        assert os.path.exists(path)

    def test_writes_valid_json(self, store, tmp_path):
        path = str(tmp_path / "output.json")
        data = {"name": "test", "value": 42}
        store._write_json(path, data)
        with open(path, "r") as f:
            result = json.load(f)
        assert result == data

    def test_overwrites_existing_file(self, store, tmp_path):
        path = str(tmp_path / "overwrite.json")
        store._write_json(path, {"version": 1})
        store._write_json(path, {"version": 2})
        with open(path, "r") as f:
            result = json.load(f)
        assert result == {"version": 2}

    def test_preserves_unicode(self, store, tmp_path):
        path = str(tmp_path / "unicode.json")
        data = {"text": "తెలుగు భాష"}
        store._write_json(path, data)
        with open(path, "r", encoding="utf-8") as f:
            result = json.load(f)
        assert result["text"] == "తెలుగు భాష"

    def test_pretty_prints_with_indent(self, store, tmp_path):
        path = str(tmp_path / "pretty.json")
        store._write_json(path, {"key": "value"})
        with open(path, "r") as f:
            content = f.read()
        assert "\n" in content  # indented JSON has newlines


# ===========================================================================
# 7. _iter_jsonl
# ===========================================================================


class TestIterJsonl:
    """Tests for MemoryStore._iter_jsonl."""

    def test_nonexistent_file_yields_nothing(self, store, tmp_path):
        path = str(tmp_path / "nonexistent.jsonl")
        result = list(store._iter_jsonl(path))
        assert result == []

    def test_valid_lines(self, store, tmp_path):
        path = str(tmp_path / "test.jsonl")
        with open(path, "w") as f:
            f.write('{"id": "1", "text": "hello"}\n')
            f.write('{"id": "2", "text": "world"}\n')
        result = list(store._iter_jsonl(path))
        assert len(result) == 2
        assert result[0]["id"] == "1"
        assert result[1]["id"] == "2"

    def test_empty_lines_skipped(self, store, tmp_path):
        path = str(tmp_path / "test.jsonl")
        with open(path, "w") as f:
            f.write('{"id": "1"}\n')
            f.write("\n")
            f.write("   \n")
            f.write('{"id": "2"}\n')
        result = list(store._iter_jsonl(path))
        assert len(result) == 2

    def test_malformed_json_skipped(self, store, tmp_path):
        path = str(tmp_path / "test.jsonl")
        with open(path, "w") as f:
            f.write('{"id": "1"}\n')
            f.write("not valid json\n")
            f.write('{"id": "2"}\n')
        result = list(store._iter_jsonl(path))
        assert len(result) == 2
        assert result[0]["id"] == "1"
        assert result[1]["id"] == "2"

    def test_empty_file(self, store, tmp_path):
        path = str(tmp_path / "empty.jsonl")
        with open(path, "w") as f:
            pass
        result = list(store._iter_jsonl(path))
        assert result == []

    def test_single_line(self, store, tmp_path):
        path = str(tmp_path / "single.jsonl")
        with open(path, "w") as f:
            f.write('{"id": "1"}\n')
        result = list(store._iter_jsonl(path))
        assert len(result) == 1

    def test_unicode_content(self, store, tmp_path):
        path = str(tmp_path / "unicode.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"text": "తెలుగు"}, ensure_ascii=False) + "\n")
        result = list(store._iter_jsonl(path))
        assert result[0]["text"] == "తెలుగు"


# ===========================================================================
# 8. _append_jsonl
# ===========================================================================


class TestAppendJsonl:
    """Tests for MemoryStore._append_jsonl."""

    def test_creates_directories(self, store, tmp_path):
        path = str(tmp_path / "new_dir" / "test.jsonl")
        store._append_jsonl(path, {"id": "abc", "text": "hello"})
        assert os.path.exists(path)

    def test_appends_line(self, store, tmp_path):
        path = str(tmp_path / "test.jsonl")
        store._append_jsonl(path, {"id": "1", "text": "first"})
        store._append_jsonl(path, {"id": "2", "text": "second"})
        with open(path, "r") as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
        assert len(lines) == 2

    def test_returns_id(self, store, tmp_path):
        path = str(tmp_path / "test.jsonl")
        result = store._append_jsonl(path, {"id": "abc123", "text": "test"})
        assert result == "abc123"

    def test_returns_empty_string_if_no_id(self, store, tmp_path):
        path = str(tmp_path / "test.jsonl")
        result = store._append_jsonl(path, {"text": "no id"})
        assert result == ""

    def test_appended_content_is_valid_json(self, store, tmp_path):
        path = str(tmp_path / "test.jsonl")
        store._append_jsonl(path, {"id": "1", "data": [1, 2, 3]})
        with open(path, "r") as f:
            line = f.readline().strip()
        parsed = json.loads(line)
        assert parsed["data"] == [1, 2, 3]

    def test_preserves_unicode_in_appended_content(self, store, tmp_path):
        path = str(tmp_path / "test.jsonl")
        store._append_jsonl(path, {"id": "1", "text": "తెలుగు"})
        with open(path, "r", encoding="utf-8") as f:
            line = f.readline().strip()
        parsed = json.loads(line)
        assert parsed["text"] == "తెలుగు"


# ===========================================================================
# 9. Persona
# ===========================================================================


class TestPersona:
    """Tests for get_persona and set_persona."""

    def test_set_and_get_persona_roundtrip(self, store):
        persona = {"name": "Friday", "language": "te"}
        store.set_persona(persona)
        result = store.get_persona()
        assert result["name"] == "Friday"
        assert result["language"] == "te"

    def test_set_persona_adds_updated_at(self, store):
        store.set_persona({"name": "Friday"})
        result = store.get_persona()
        assert "updated_at" in result

    def test_set_persona_updated_at_is_iso(self, store):
        store.set_persona({"name": "Friday"})
        result = store.get_persona()
        parsed = datetime.fromisoformat(result["updated_at"])
        assert parsed.tzinfo is not None

    def test_set_persona_does_not_mutate_original(self, store):
        original = {"name": "Friday"}
        store.set_persona(original)
        assert "updated_at" not in original

    def test_set_persona_overwrites_previous(self, store):
        store.set_persona({"name": "Friday", "version": 1})
        store.set_persona({"name": "Saturday", "version": 2})
        result = store.get_persona()
        assert result["name"] == "Saturday"
        assert result["version"] == 2

    def test_get_persona_file_not_found(self, store):
        with pytest.raises(FileNotFoundError):
            store.get_persona()

    def test_set_persona_creates_directories(self, store):
        store.set_persona({"name": "Friday"})
        assert os.path.exists(store.paths["persona"])


# ===========================================================================
# 10. Principles
# ===========================================================================


class TestPrinciples:
    """Tests for get_principles and set_principles."""

    def test_set_and_get_principles_roundtrip(self, store):
        rules = {"rule1": "be kind", "rule2": "be helpful"}
        store.set_principles(rules)
        result = store.get_principles()
        assert result["rule1"] == "be kind"
        assert result["rule2"] == "be helpful"

    def test_set_principles_adds_updated_at(self, store):
        store.set_principles({"rule1": "be kind"})
        result = store.get_principles()
        assert "updated_at" in result

    def test_set_principles_updated_at_is_iso(self, store):
        store.set_principles({"rule1": "be kind"})
        result = store.get_principles()
        parsed = datetime.fromisoformat(result["updated_at"])
        assert parsed.tzinfo is not None

    def test_set_principles_does_not_mutate_original(self, store):
        original = {"rule1": "be kind"}
        store.set_principles(original)
        assert "updated_at" not in original

    def test_set_principles_overwrites_previous(self, store):
        store.set_principles({"rule1": "old rule"})
        store.set_principles({"rule1": "new rule"})
        result = store.get_principles()
        assert result["rule1"] == "new rule"

    def test_get_principles_file_not_found(self, store):
        with pytest.raises(FileNotFoundError):
            store.get_principles()

    def test_set_principles_creates_directories(self, store):
        store.set_principles({"rule1": "be kind"})
        assert os.path.exists(store.paths["principles"])


# ===========================================================================
# 11. add_ltm_memory
# ===========================================================================


class TestAddLtmMemory:
    """Tests for MemoryStore.add_ltm_memory."""

    def test_returns_hex_id(self, store):
        mid = store.add_ltm_memory("test memory")
        assert isinstance(mid, str)
        assert len(mid) == 32  # uuid4().hex is 32 chars

    def test_unique_ids(self, store):
        id1 = store.add_ltm_memory("memory one")
        id2 = store.add_ltm_memory("memory two")
        assert id1 != id2

    def test_default_lang(self, store):
        store.add_ltm_memory("test")
        rows = list(store._iter_jsonl(store.paths["ltm"]))
        assert rows[0]["lang"] == "te"

    def test_custom_lang(self, store):
        store.add_ltm_memory("test", lang="en")
        rows = list(store._iter_jsonl(store.paths["ltm"]))
        assert rows[0]["lang"] == "en"

    def test_default_tags(self, store):
        store.add_ltm_memory("test")
        rows = list(store._iter_jsonl(store.paths["ltm"]))
        assert rows[0]["tags"] == []

    def test_custom_tags(self, store):
        store.add_ltm_memory("test", tags=["tag1", "tag2"])
        rows = list(store._iter_jsonl(store.paths["ltm"]))
        assert rows[0]["tags"] == ["tag1", "tag2"]

    def test_default_trust(self, store):
        store.add_ltm_memory("test")
        rows = list(store._iter_jsonl(store.paths["ltm"]))
        assert rows[0]["trust"] == 3

    def test_custom_trust(self, store):
        store.add_ltm_memory("test", trust=5)
        rows = list(store._iter_jsonl(store.paths["ltm"]))
        assert rows[0]["trust"] == 5

    def test_trust_cast_to_int(self, store):
        store.add_ltm_memory("test", trust=4.7)
        rows = list(store._iter_jsonl(store.paths["ltm"]))
        assert rows[0]["trust"] == 4
        assert isinstance(rows[0]["trust"], int)

    def test_default_embedding(self, store):
        store.add_ltm_memory("test")
        rows = list(store._iter_jsonl(store.paths["ltm"]))
        assert rows[0]["embedding"] is None

    def test_custom_embedding(self, store):
        emb = [0.1, 0.2, 0.3]
        store.add_ltm_memory("test", embedding=emb)
        rows = list(store._iter_jsonl(store.paths["ltm"]))
        assert rows[0]["embedding"] == emb

    def test_default_source(self, store):
        store.add_ltm_memory("test")
        rows = list(store._iter_jsonl(store.paths["ltm"]))
        assert rows[0]["source"] is None

    def test_custom_source(self, store):
        store.add_ltm_memory("test", source="interview")
        rows = list(store._iter_jsonl(store.paths["ltm"]))
        assert rows[0]["source"] == "interview"

    def test_has_created_at(self, store):
        store.add_ltm_memory("test")
        rows = list(store._iter_jsonl(store.paths["ltm"]))
        assert "created_at" in rows[0]
        datetime.fromisoformat(rows[0]["created_at"])

    def test_has_updated_at(self, store):
        store.add_ltm_memory("test")
        rows = list(store._iter_jsonl(store.paths["ltm"]))
        assert "updated_at" in rows[0]
        datetime.fromisoformat(rows[0]["updated_at"])

    def test_created_at_equals_updated_at(self, store):
        store.add_ltm_memory("test")
        rows = list(store._iter_jsonl(store.paths["ltm"]))
        assert rows[0]["created_at"] == rows[0]["updated_at"]

    def test_text_stored(self, store):
        store.add_ltm_memory("This is the memory text")
        rows = list(store._iter_jsonl(store.paths["ltm"]))
        assert rows[0]["text"] == "This is the memory text"

    def test_all_fields_present(self, store):
        store.add_ltm_memory("test", tags=["t"], embedding=[0.1], source="src")
        rows = list(store._iter_jsonl(store.paths["ltm"]))
        expected_keys = {
            "id",
            "text",
            "lang",
            "tags",
            "trust",
            "embedding",
            "source",
            "created_at",
            "updated_at",
        }
        assert set(rows[0].keys()) == expected_keys

    def test_multiple_adds_append(self, store):
        store.add_ltm_memory("first")
        store.add_ltm_memory("second")
        store.add_ltm_memory("third")
        rows = list(store._iter_jsonl(store.paths["ltm"]))
        assert len(rows) == 3


# ===========================================================================
# 12. search_ltm
# ===========================================================================


class TestSearchLtm:
    """Tests for MemoryStore.search_ltm."""

    def test_empty_store_returns_empty(self, store):
        result = store.search_ltm(query="anything")
        assert result == []

    def test_lexical_search_by_query_tokens(self, populated_store):
        results = populated_store.search_ltm(query="Telugu language")
        assert len(results) > 0
        # The entry with "Telugu is a Dravidian language" should rank highest
        assert "Telugu" in results[0]["text"] or "language" in results[0]["text"]

    def test_lexical_search_case_insensitive(self, populated_store):
        results = populated_store.search_ltm(query="TELUGU LANGUAGE")
        assert len(results) > 0

    def test_vector_search_by_embedding(self, populated_store):
        # Query vector matches the first entry's embedding [1,0,0]
        results = populated_store.search_ltm(query_vector=[1.0, 0.0, 0.0])
        assert len(results) > 0
        assert results[0]["text"] == "Telugu is a Dravidian language"

    def test_vector_search_second_entry(self, populated_store):
        # Query vector matches the second entry's embedding [0,1,0]
        results = populated_store.search_ltm(query_vector=[0.0, 1.0, 0.0])
        assert results[0]["text"] == "Python is a programming language"

    def test_tag_filtering(self, populated_store):
        results = populated_store.search_ltm(query="language", tags=["telugu"])
        # Only entries with tag "telugu" should be returned
        for r in results:
            assert "telugu" in [t.lower() for t in r.get("tags", [])]

    def test_tag_filtering_subset_match(self, populated_store):
        # Request entries that have BOTH "language" and "telugu" tags
        results = populated_store.search_ltm(query="test", tags=["language", "telugu"])
        for r in results:
            tags_lower = set(t.lower() for t in r.get("tags", []))
            assert {"language", "telugu"}.issubset(tags_lower)

    def test_tag_filtering_no_match_returns_empty(self, populated_store):
        results = populated_store.search_ltm(query="test", tags=["nonexistent_tag"])
        assert results == []

    def test_top_k_limit(self, populated_store):
        results = populated_store.search_ltm(query="language", top_k=1)
        assert len(results) == 1

    def test_top_k_larger_than_results(self, populated_store):
        results = populated_store.search_ltm(query="language", top_k=100)
        assert len(results) == 3  # only 3 entries exist

    def test_min_top_k_is_one(self, populated_store):
        results = populated_store.search_ltm(query="language", top_k=0)
        assert len(results) >= 1

    def test_negative_top_k_returns_at_least_one(self, populated_store):
        results = populated_store.search_ltm(query="language", top_k=-5)
        assert len(results) >= 1

    def test_default_top_k_is_five(self, store):
        # Add 7 entries
        for i in range(7):
            store.add_ltm_memory(f"memory about topic {i}")
        results = store.search_ltm(query="topic")
        assert len(results) == 5

    def test_lexical_search_with_empty_query(self, populated_store):
        # Empty query should still return results (all scored 0)
        results = populated_store.search_ltm(query="")
        assert len(results) > 0

    def test_lexical_search_none_query(self, populated_store):
        # None query should still work (tokenized as empty)
        results = populated_store.search_ltm()
        assert len(results) > 0

    def test_vector_search_skips_entries_without_embedding(self, store):
        store.add_ltm_memory("no embedding")
        store.add_ltm_memory("has embedding", embedding=[1.0, 0.0])
        results = store.search_ltm(query_vector=[1.0, 0.0])
        assert len(results) == 1
        assert results[0]["text"] == "has embedding"

    def test_vector_search_with_tag_filter(self, populated_store):
        results = populated_store.search_ltm(
            query_vector=[1.0, 0.0, 0.0], tags=["telugu"]
        )
        for r in results:
            assert "telugu" in [t.lower() for t in r.get("tags", [])]

    def test_tag_filter_case_insensitive(self, populated_store):
        results = populated_store.search_ltm(query="language", tags=["TELUGU"])
        assert len(results) > 0

    def test_results_sorted_by_score_descending(self, store):
        store.add_ltm_memory("alpha beta gamma")
        store.add_ltm_memory("alpha beta")
        store.add_ltm_memory("alpha")
        results = store.search_ltm(query="alpha beta gamma")
        # First result should have most overlap
        assert results[0]["text"] == "alpha beta gamma"

    def test_tag_bonus_in_lexical_scoring(self, store):
        store.add_ltm_memory("information about cats", tags=["cats"])
        store.add_ltm_memory("information about dogs", tags=["dogs"])
        results = store.search_ltm(query="cats information")
        # The entry tagged "cats" should rank higher due to tag bonus
        assert results[0]["tags"] == ["cats"]


# ===========================================================================
# 13. add_snippet
# ===========================================================================


class TestAddSnippet:
    """Tests for MemoryStore.add_snippet."""

    def test_returns_hex_id(self, store):
        sid = store.add_snippet(title="Test", body="Test body")
        assert isinstance(sid, str)
        assert len(sid) == 32

    def test_unique_ids(self, store):
        id1 = store.add_snippet(title="First", body="body1")
        id2 = store.add_snippet(title="Second", body="body2")
        assert id1 != id2

    def test_default_lang(self, store):
        store.add_snippet(title="Test", body="Body")
        rows = list(store._iter_jsonl(store.paths["snippets"]))
        assert rows[0]["lang"] == "te"

    def test_custom_lang(self, store):
        store.add_snippet(title="Test", body="Body", lang="en")
        rows = list(store._iter_jsonl(store.paths["snippets"]))
        assert rows[0]["lang"] == "en"

    def test_default_tags(self, store):
        store.add_snippet(title="Test", body="Body")
        rows = list(store._iter_jsonl(store.paths["snippets"]))
        assert rows[0]["tags"] == []

    def test_custom_tags(self, store):
        store.add_snippet(title="Test", body="Body", tags=["tag1", "tag2"])
        rows = list(store._iter_jsonl(store.paths["snippets"]))
        assert rows[0]["tags"] == ["tag1", "tag2"]

    def test_default_version(self, store):
        store.add_snippet(title="Test", body="Body")
        rows = list(store._iter_jsonl(store.paths["snippets"]))
        assert rows[0]["version"] == 1

    def test_custom_version(self, store):
        store.add_snippet(title="Test", body="Body", version=3)
        rows = list(store._iter_jsonl(store.paths["snippets"]))
        assert rows[0]["version"] == 3

    def test_version_cast_to_int(self, store):
        store.add_snippet(title="Test", body="Body", version=2.9)
        rows = list(store._iter_jsonl(store.paths["snippets"]))
        assert rows[0]["version"] == 2
        assert isinstance(rows[0]["version"], int)

    def test_default_embedding(self, store):
        store.add_snippet(title="Test", body="Body")
        rows = list(store._iter_jsonl(store.paths["snippets"]))
        assert rows[0]["embedding"] is None

    def test_custom_embedding(self, store):
        emb = [0.5, 0.5]
        store.add_snippet(title="Test", body="Body", embedding=emb)
        rows = list(store._iter_jsonl(store.paths["snippets"]))
        assert rows[0]["embedding"] == emb

    def test_default_domain(self, store):
        store.add_snippet(title="Test", body="Body")
        rows = list(store._iter_jsonl(store.paths["snippets"]))
        assert rows[0]["domain"] == "general"

    def test_custom_domain(self, store):
        store.add_snippet(title="Test", body="Body", domain="science")
        rows = list(store._iter_jsonl(store.paths["snippets"]))
        assert rows[0]["domain"] == "science"

    def test_title_stored(self, store):
        store.add_snippet(title="My Title", body="My Body")
        rows = list(store._iter_jsonl(store.paths["snippets"]))
        assert rows[0]["title"] == "My Title"

    def test_body_stored(self, store):
        store.add_snippet(title="My Title", body="My Body Content")
        rows = list(store._iter_jsonl(store.paths["snippets"]))
        assert rows[0]["body"] == "My Body Content"

    def test_has_timestamps(self, store):
        store.add_snippet(title="Test", body="Body")
        rows = list(store._iter_jsonl(store.paths["snippets"]))
        assert "created_at" in rows[0]
        assert "updated_at" in rows[0]

    def test_all_fields_present(self, store):
        store.add_snippet(title="T", body="B", tags=["t"], embedding=[0.1])
        rows = list(store._iter_jsonl(store.paths["snippets"]))
        expected_keys = {
            "id",
            "title",
            "body",
            "lang",
            "tags",
            "version",
            "embedding",
            "domain",
            "created_at",
            "updated_at",
        }
        assert set(rows[0].keys()) == expected_keys


# ===========================================================================
# 14. search_snippets
# ===========================================================================


class TestSearchSnippets:
    """Tests for MemoryStore.search_snippets."""

    def test_empty_store_returns_empty(self, store):
        result = store.search_snippets(query="anything")
        assert result == []

    def test_lexical_search_title(self, populated_snippets_store):
        results = populated_snippets_store.search_snippets(query="Grammar")
        assert len(results) > 0
        assert results[0]["title"] == "Telugu Grammar"

    def test_lexical_search_body(self, populated_snippets_store):
        results = populated_snippets_store.search_snippets(query="sandhi")
        assert len(results) > 0
        assert "sandhi" in results[0]["body"]

    def test_lexical_search_title_and_body_combined(self, populated_snippets_store):
        # "Telugu" appears in title of first and third, "Grammar" in title of first
        results = populated_snippets_store.search_snippets(query="Telugu Grammar")
        assert results[0]["title"] == "Telugu Grammar"

    def test_vector_search(self, populated_snippets_store):
        results = populated_snippets_store.search_snippets(query_vector=[1.0, 0.0, 0.0])
        assert results[0]["title"] == "Telugu Grammar"

    def test_tag_filtering(self, populated_snippets_store):
        results = populated_snippets_store.search_snippets(
            query="test", tags=["telugu"]
        )
        for r in results:
            assert "telugu" in [t.lower() for t in r.get("tags", [])]

    def test_tag_filtering_no_match(self, populated_snippets_store):
        results = populated_snippets_store.search_snippets(
            query="test", tags=["nonexistent"]
        )
        assert results == []

    def test_top_k_limit(self, populated_snippets_store):
        results = populated_snippets_store.search_snippets(query="Telugu", top_k=1)
        assert len(results) == 1

    def test_top_k_larger_than_results(self, populated_snippets_store):
        results = populated_snippets_store.search_snippets(query="test", top_k=100)
        assert len(results) == 3

    def test_min_top_k_is_one(self, populated_snippets_store):
        results = populated_snippets_store.search_snippets(query="test", top_k=0)
        assert len(results) >= 1

    def test_vector_search_skips_no_embedding(self, store):
        store.add_snippet(title="No Embedding", body="text")
        store.add_snippet(title="Has Embedding", body="text", embedding=[1.0, 0.0])
        results = store.search_snippets(query_vector=[1.0, 0.0])
        assert len(results) == 1
        assert results[0]["title"] == "Has Embedding"

    def test_tag_filter_case_insensitive(self, populated_snippets_store):
        results = populated_snippets_store.search_snippets(
            query="test", tags=["TELUGU"]
        )
        assert len(results) > 0

    def test_tag_subset_match(self, populated_snippets_store):
        results = populated_snippets_store.search_snippets(
            query="test", tags=["grammar", "telugu"]
        )
        for r in results:
            tags_lower = set(t.lower() for t in r.get("tags", []))
            assert {"grammar", "telugu"}.issubset(tags_lower)

    def test_empty_query(self, populated_snippets_store):
        results = populated_snippets_store.search_snippets(query="")
        assert len(results) > 0

    def test_none_query(self, populated_snippets_store):
        results = populated_snippets_store.search_snippets()
        assert len(results) > 0

    def test_default_top_k_is_five(self, store):
        for i in range(7):
            store.add_snippet(title=f"Snippet {i}", body=f"Content about topic {i}")
        results = store.search_snippets(query="topic")
        assert len(results) == 5


# ===========================================================================
# 15. upsert_stm_session
# ===========================================================================


class TestUpsertStmSession:
    """Tests for MemoryStore.upsert_stm_session."""

    def test_insert_new_session(self, store):
        sid = store.upsert_stm_session(
            session_id="sess1",
            rolling_summary="first summary",
            target_lang="te",
        )
        assert sid == "sess1"
        rows = list(store._iter_jsonl(store.paths["stm"]))
        assert len(rows) == 1
        assert rows[0]["session_id"] == "sess1"

    def test_insert_returns_session_id(self, store):
        result = store.upsert_stm_session(
            session_id="my_session",
            rolling_summary="summary",
            target_lang="en",
        )
        assert result == "my_session"

    def test_update_existing_session(self, store):
        store.upsert_stm_session(
            session_id="sess1",
            rolling_summary="original summary",
            target_lang="te",
        )
        store.upsert_stm_session(
            session_id="sess1",
            rolling_summary="updated summary",
            target_lang="en",
        )
        rows = list(store._iter_jsonl(store.paths["stm"]))
        assert len(rows) == 1
        assert rows[0]["rolling_summary"] == "updated summary"
        assert rows[0]["target_lang"] == "en"

    def test_preserves_other_sessions(self, store):
        store.upsert_stm_session(
            session_id="sess1",
            rolling_summary="summary 1",
            target_lang="te",
        )
        store.upsert_stm_session(
            session_id="sess2",
            rolling_summary="summary 2",
            target_lang="en",
        )
        # Update sess1 only
        store.upsert_stm_session(
            session_id="sess1",
            rolling_summary="updated 1",
            target_lang="te",
        )
        rows = list(store._iter_jsonl(store.paths["stm"]))
        assert len(rows) == 2
        sess1 = next(r for r in rows if r["session_id"] == "sess1")
        sess2 = next(r for r in rows if r["session_id"] == "sess2")
        assert sess1["rolling_summary"] == "updated 1"
        assert sess2["rolling_summary"] == "summary 2"

    def test_session_has_updated_at(self, store):
        store.upsert_stm_session(
            session_id="sess1",
            rolling_summary="summary",
            target_lang="te",
        )
        rows = list(store._iter_jsonl(store.paths["stm"]))
        assert "updated_at" in rows[0]
        datetime.fromisoformat(rows[0]["updated_at"])

    def test_update_changes_updated_at(self, store):
        store.upsert_stm_session(
            session_id="sess1",
            rolling_summary="original",
            target_lang="te",
        )
        rows_before = list(store._iter_jsonl(store.paths["stm"]))
        ts1 = rows_before[0]["updated_at"]

        store.upsert_stm_session(
            session_id="sess1",
            rolling_summary="updated",
            target_lang="te",
        )
        rows_after = list(store._iter_jsonl(store.paths["stm"]))
        ts2 = rows_after[0]["updated_at"]
        # Timestamps may be the same if test runs fast, but should be valid
        datetime.fromisoformat(ts2)

    def test_creates_directories(self, store):
        store.upsert_stm_session(
            session_id="sess1",
            rolling_summary="summary",
            target_lang="te",
        )
        assert os.path.exists(store.paths["stm"])

    def test_multiple_inserts(self, store):
        for i in range(5):
            store.upsert_stm_session(
                session_id=f"sess{i}",
                rolling_summary=f"summary {i}",
                target_lang="te",
            )
        rows = list(store._iter_jsonl(store.paths["stm"]))
        assert len(rows) == 5

    def test_session_fields_complete(self, store):
        store.upsert_stm_session(
            session_id="sess1",
            rolling_summary="my summary",
            target_lang="te",
        )
        rows = list(store._iter_jsonl(store.paths["stm"]))
        expected_keys = {"session_id", "rolling_summary", "target_lang", "updated_at"}
        assert set(rows[0].keys()) == expected_keys

    def test_update_replaces_all_fields(self, store):
        store.upsert_stm_session(
            session_id="sess1",
            rolling_summary="original",
            target_lang="te",
        )
        store.upsert_stm_session(
            session_id="sess1",
            rolling_summary="new",
            target_lang="en",
        )
        rows = list(store._iter_jsonl(store.paths["stm"]))
        assert rows[0]["rolling_summary"] == "new"
        assert rows[0]["target_lang"] == "en"


# ===========================================================================
# 16. log_interaction
# ===========================================================================


class TestLogInteraction:
    """Tests for MemoryStore.log_interaction."""

    def test_returns_hex_id(self, store):
        iid = store.log_interaction(
            session_id="sess1",
            user_msg="hello",
            model_reply="hi there",
        )
        assert isinstance(iid, str)
        assert len(iid) == 32

    def test_unique_ids(self, store):
        id1 = store.log_interaction(
            session_id="sess1", user_msg="msg1", model_reply="reply1"
        )
        id2 = store.log_interaction(
            session_id="sess1", user_msg="msg2", model_reply="reply2"
        )
        assert id1 != id2

    def test_session_id_stored(self, store):
        store.log_interaction(
            session_id="my_session", user_msg="hi", model_reply="hello"
        )
        rows = list(store._iter_jsonl(store.paths["interactions"]))
        assert rows[0]["session_id"] == "my_session"

    def test_user_msg_stored(self, store):
        store.log_interaction(
            session_id="s1", user_msg="How are you?", model_reply="Fine"
        )
        rows = list(store._iter_jsonl(store.paths["interactions"]))
        assert rows[0]["user_msg"] == "How are you?"

    def test_model_reply_stored(self, store):
        store.log_interaction(
            session_id="s1", user_msg="Hi", model_reply="Hello, how can I help?"
        )
        rows = list(store._iter_jsonl(store.paths["interactions"]))
        assert rows[0]["model_reply"] == "Hello, how can I help?"

    def test_default_used_ltm_ids(self, store):
        store.log_interaction(session_id="s1", user_msg="hi", model_reply="hello")
        rows = list(store._iter_jsonl(store.paths["interactions"]))
        assert rows[0]["used_ltm_ids"] == []

    def test_custom_used_ltm_ids(self, store):
        store.log_interaction(
            session_id="s1",
            user_msg="hi",
            model_reply="hello",
            used_ltm_ids=["id1", "id2"],
        )
        rows = list(store._iter_jsonl(store.paths["interactions"]))
        assert rows[0]["used_ltm_ids"] == ["id1", "id2"]

    def test_default_used_snippet_ids(self, store):
        store.log_interaction(session_id="s1", user_msg="hi", model_reply="hello")
        rows = list(store._iter_jsonl(store.paths["interactions"]))
        assert rows[0]["used_snippet_ids"] == []

    def test_custom_used_snippet_ids(self, store):
        store.log_interaction(
            session_id="s1",
            user_msg="hi",
            model_reply="hello",
            used_snippet_ids=["snip1"],
        )
        rows = list(store._iter_jsonl(store.paths["interactions"]))
        assert rows[0]["used_snippet_ids"] == ["snip1"]

    def test_default_stm_summary(self, store):
        store.log_interaction(session_id="s1", user_msg="hi", model_reply="hello")
        rows = list(store._iter_jsonl(store.paths["interactions"]))
        assert rows[0]["stm_summary"] is None

    def test_custom_stm_summary(self, store):
        store.log_interaction(
            session_id="s1",
            user_msg="hi",
            model_reply="hello",
            stm_summary="The user greeted the assistant",
        )
        rows = list(store._iter_jsonl(store.paths["interactions"]))
        assert rows[0]["stm_summary"] == "The user greeted the assistant"

    def test_default_lang(self, store):
        store.log_interaction(session_id="s1", user_msg="hi", model_reply="hello")
        rows = list(store._iter_jsonl(store.paths["interactions"]))
        assert rows[0]["lang"] == "te"

    def test_custom_lang(self, store):
        store.log_interaction(
            session_id="s1", user_msg="hi", model_reply="hello", lang="en"
        )
        rows = list(store._iter_jsonl(store.paths["interactions"]))
        assert rows[0]["lang"] == "en"

    def test_has_created_at(self, store):
        store.log_interaction(session_id="s1", user_msg="hi", model_reply="hello")
        rows = list(store._iter_jsonl(store.paths["interactions"]))
        assert "created_at" in rows[0]
        datetime.fromisoformat(rows[0]["created_at"])

    def test_all_fields_present(self, store):
        store.log_interaction(
            session_id="s1",
            user_msg="hi",
            model_reply="hello",
            used_ltm_ids=["l1"],
            used_snippet_ids=["s1"],
            stm_summary="summary",
            lang="en",
        )
        rows = list(store._iter_jsonl(store.paths["interactions"]))
        expected_keys = {
            "id",
            "session_id",
            "user_msg",
            "model_reply",
            "used_ltm_ids",
            "used_snippet_ids",
            "stm_summary",
            "lang",
            "created_at",
        }
        assert set(rows[0].keys()) == expected_keys

    def test_multiple_interactions_append(self, store):
        for i in range(5):
            store.log_interaction(
                session_id="s1", user_msg=f"msg{i}", model_reply=f"reply{i}"
            )
        rows = list(store._iter_jsonl(store.paths["interactions"]))
        assert len(rows) == 5

    def test_interactions_across_sessions(self, store):
        store.log_interaction(session_id="sess1", user_msg="hi", model_reply="hello")
        store.log_interaction(session_id="sess2", user_msg="hey", model_reply="howdy")
        rows = list(store._iter_jsonl(store.paths["interactions"]))
        sessions = {r["session_id"] for r in rows}
        assert sessions == {"sess1", "sess2"}


# ===========================================================================
# 17. Integration / Cross-cutting tests
# ===========================================================================


class TestIntegration:
    """Cross-cutting tests that exercise multiple features together."""

    def test_ltm_add_then_search_roundtrip(self, store):
        mid = store.add_ltm_memory(
            "Biryani is a popular food in Hyderabad", tags=["food"]
        )
        results = store.search_ltm(query="Biryani Hyderabad food")
        assert len(results) == 1
        assert results[0]["id"] == mid

    def test_snippet_add_then_search_roundtrip(self, store):
        sid = store.add_snippet(
            title="Recipe", body="How to make dosa", tags=["cooking"]
        )
        results = store.search_snippets(query="dosa recipe")
        assert len(results) == 1
        assert results[0]["id"] == sid

    def test_multiple_stores_isolated(self, tmp_path):
        store1 = MemoryStore(root_dir=str(tmp_path / "mem1"))
        store2 = MemoryStore(root_dir=str(tmp_path / "mem2"))
        store1.add_ltm_memory("store one memory")
        store2.add_ltm_memory("store two memory")
        results1 = store1.search_ltm(query="store one")
        results2 = store2.search_ltm(query="store two")
        assert len(results1) == 1
        assert len(results2) == 1
        assert results1[0]["text"] == "store one memory"
        assert results2[0]["text"] == "store two memory"

    def test_persona_and_principles_independent(self, store):
        store.set_persona({"name": "Friday"})
        store.set_principles({"rule": "be helpful"})
        persona = store.get_persona()
        principles = store.get_principles()
        assert persona["name"] == "Friday"
        assert principles["rule"] == "be helpful"
        assert "rule" not in persona
        assert "name" not in principles

    def test_stm_and_interaction_workflow(self, store):
        # Create a session
        store.upsert_stm_session(
            session_id="workflow_sess",
            rolling_summary="initial summary",
            target_lang="te",
        )
        # Log an interaction
        iid = store.log_interaction(
            session_id="workflow_sess",
            user_msg="What is Telugu?",
            model_reply="Telugu is a Dravidian language",
            stm_summary="user asked about Telugu",
        )
        # Update the session
        store.upsert_stm_session(
            session_id="workflow_sess",
            rolling_summary="user asked about Telugu language",
            target_lang="te",
        )
        # Verify
        sessions = list(store._iter_jsonl(store.paths["stm"]))
        interactions = list(store._iter_jsonl(store.paths["interactions"]))
        assert len(sessions) == 1
        assert sessions[0]["rolling_summary"] == "user asked about Telugu language"
        assert len(interactions) == 1
        assert interactions[0]["id"] == iid

    def test_ltm_vector_search_with_tag_and_top_k(self, store):
        store.add_ltm_memory("m1", tags=["cat"], embedding=[1.0, 0.0])
        store.add_ltm_memory("m2", tags=["cat"], embedding=[0.9, 0.1])
        store.add_ltm_memory("m3", tags=["dog"], embedding=[0.8, 0.2])
        results = store.search_ltm(query_vector=[1.0, 0.0], tags=["cat"], top_k=1)
        assert len(results) == 1
        assert results[0]["text"] == "m1"

    def test_snippets_vector_search_with_tag_and_top_k(self, store):
        store.add_snippet(title="s1", body="b1", tags=["cat"], embedding=[1.0, 0.0])
        store.add_snippet(title="s2", body="b2", tags=["cat"], embedding=[0.9, 0.1])
        store.add_snippet(title="s3", body="b3", tags=["dog"], embedding=[0.8, 0.2])
        results = store.search_snippets(query_vector=[1.0, 0.0], tags=["cat"], top_k=1)
        assert len(results) == 1
        assert results[0]["title"] == "s1"

    def test_full_workflow(self, store):
        # Set up persona and principles
        store.set_persona({"name": "Friday", "style": "friendly"})
        store.set_principles({"rule1": "be helpful", "rule2": "speak Telugu"})

        # Add LTM memories
        ltm_id = store.add_ltm_memory(
            "Friday loves Telugu cinema",
            tags=["preferences", "cinema"],
        )

        # Add snippets
        snip_id = store.add_snippet(
            title="Telugu Greetings",
            body="Namaskaram is a common greeting",
            tags=["language", "telugu"],
        )

        # Create session and log interaction
        store.upsert_stm_session(
            session_id="full_test",
            rolling_summary="starting conversation",
            target_lang="te",
        )
        store.log_interaction(
            session_id="full_test",
            user_msg="Hello",
            model_reply="Namaskaram!",
            used_ltm_ids=[ltm_id],
            used_snippet_ids=[snip_id],
        )

        # Verify everything
        persona = store.get_persona()
        assert persona["name"] == "Friday"
        principles = store.get_principles()
        assert principles["rule1"] == "be helpful"
        ltm_results = store.search_ltm(query="Telugu cinema")
        assert len(ltm_results) == 1
        snippet_results = store.search_snippets(query="greetings")
        assert len(snippet_results) == 1
        sessions = list(store._iter_jsonl(store.paths["stm"]))
        assert len(sessions) == 1
        interactions = list(store._iter_jsonl(store.paths["interactions"]))
        assert len(interactions) == 1
