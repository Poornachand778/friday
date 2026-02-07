"""MCP Document Processing Tools for Friday AI"""

from mcp.documents.service import (
    # Document management
    document_ingest,
    document_search,
    document_get_context,
    document_list,
    document_get,
    document_get_chapter,
    document_status,
    document_delete,
    # Inbox sync
    inbox_scan,
    inbox_list,
    inbox_status,
    inbox_start_watch,
    inbox_stop_watch,
    # Book understanding
    book_study,
    book_study_status,
    book_study_jobs,
    book_list_studied,
    book_get_understanding,
    # Mentor tools
    mentor_load_books,
    mentor_analyze,
    mentor_brainstorm,
    mentor_check_rules,
    mentor_find_inspiration,
    mentor_ask,
    mentor_compare,
    # Knowledge search
    knowledge_search,
)

__all__ = [
    # Document management
    "document_ingest",
    "document_search",
    "document_get_context",
    "document_list",
    "document_get",
    "document_get_chapter",
    "document_status",
    "document_delete",
    # Inbox sync
    "inbox_scan",
    "inbox_list",
    "inbox_status",
    "inbox_start_watch",
    "inbox_stop_watch",
    # Book understanding
    "book_study",
    "book_study_status",
    "book_study_jobs",
    "book_list_studied",
    "book_get_understanding",
    # Mentor tools
    "mentor_load_books",
    "mentor_analyze",
    "mentor_brainstorm",
    "mentor_check_rules",
    "mentor_find_inspiration",
    "mentor_ask",
    "mentor_compare",
    # Knowledge search
    "knowledge_search",
]
