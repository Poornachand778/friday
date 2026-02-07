"""Document storage layer"""

from documents.storage.document_store import DocumentStore
from documents.storage.understanding_store import BookUnderstandingStore

__all__ = ["DocumentStore", "BookUnderstandingStore"]
