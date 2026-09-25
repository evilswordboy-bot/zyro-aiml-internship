"""
SQLite Database Layer for AI Document Intelligence & Workflow Platform.
Provides robust schema management, parameterized queries, and full CRUD operations.
"""

import os
import sqlite3
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
from contextlib import contextmanager

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DB_PATH = os.path.join(BASE_DIR, "data", "documents.db")


class DatabaseManager:
    """Manages SQLite connection lifecycle, migrations, and CRUD operations."""

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.init_db()

    @contextmanager
    def _connection(self):
        """Context manager that ensures the SQLite connection is closed on exit."""
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        try:
            yield conn
        finally:
            conn.close()

    def init_db(self) -> None:
        """Initializes the database schema and indices."""
        with self._connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    original_filename TEXT NOT NULL,
                    stored_filename TEXT NOT NULL,
                    document_type TEXT NOT NULL,
                    upload_date TEXT NOT NULL,
                    company TEXT DEFAULT 'Not Found',
                    invoice_number TEXT DEFAULT 'Not Found',
                    total_amount TEXT DEFAULT 'Not Found',
                    candidate_name TEXT DEFAULT 'Not Found',
                    candidate_email TEXT DEFAULT 'Not Found',
                    candidate_phone TEXT DEFAULT 'Not Found',
                    candidate_skills TEXT DEFAULT 'Not Found',
                    file_path TEXT NOT NULL,
                    text_preview TEXT NOT NULL,
                    file_hash TEXT NOT NULL UNIQUE,
                    status TEXT NOT NULL,
                    status_reason TEXT DEFAULT '',
                    metadata_json TEXT DEFAULT '{}'
                );
            """)

            # Create performance indices for fast lookups & multi-field queries
            conn.execute("CREATE INDEX IF NOT EXISTS idx_doc_hash ON documents(file_hash);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_doc_type ON documents(document_type);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_doc_status ON documents(status);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_doc_upload_date ON documents(upload_date);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_doc_company ON documents(company);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_doc_inv_num ON documents(invoice_number);")
            conn.commit()

    def add_document(self, doc_data: Dict[str, Any]) -> int:
        """
        Inserts a newly processed document record.
        Uses parameterized queries to prevent SQL injection.
        """
        upload_date = doc_data.get("upload_date") or datetime.now().isoformat()
        metadata_json = doc_data.get("metadata_json")
        if isinstance(metadata_json, dict):
            metadata_json = json.dumps(metadata_json)
        elif not metadata_json:
            metadata_json = "{}"

        query = """
            INSERT INTO documents (
                original_filename, stored_filename, document_type, upload_date,
                company, invoice_number, total_amount, candidate_name,
                candidate_email, candidate_phone, candidate_skills,
                file_path, text_preview, file_hash, status, status_reason, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        params = (
            doc_data.get("original_filename", "untitled"),
            doc_data.get("stored_filename", "untitled"),
            doc_data.get("document_type", "Other"),
            upload_date,
            doc_data.get("company", "Not Found"),
            doc_data.get("invoice_number", "Not Found"),
            doc_data.get("total_amount", "Not Found"),
            doc_data.get("candidate_name", "Not Found"),
            doc_data.get("candidate_email", "Not Found"),
            doc_data.get("candidate_phone", "Not Found"),
            doc_data.get("candidate_skills", "Not Found"),
            doc_data.get("file_path", ""),
            doc_data.get("text_preview", "")[:1000],
            doc_data.get("file_hash", ""),
            doc_data.get("status", "Processed"),
            doc_data.get("status_reason", ""),
            metadata_json
        )

        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            conn.commit()
            return cursor.lastrowid

    def get_document_by_id(self, doc_id: int) -> Optional[Dict[str, Any]]:
        """Retrieves a single document record by primary key."""
        query = "SELECT * FROM documents WHERE id = ?;"
        with self._connection() as conn:
            row = conn.execute(query, (doc_id,)).fetchone()
            if row:
                doc = dict(row)
                try:
                    doc["metadata"] = json.loads(doc.get("metadata_json", "{}"))
                except Exception:
                    doc["metadata"] = {}
                return doc
            return None

    def get_document_by_hash(self, file_hash: str) -> Optional[Dict[str, Any]]:
        """Retrieves a document record by SHA-256 hash (duplicate detection)."""
        query = "SELECT * FROM documents WHERE file_hash = ?;"
        with self._connection() as conn:
            row = conn.execute(query, (file_hash,)).fetchone()
            if row:
                doc = dict(row)
                try:
                    doc["metadata"] = json.loads(doc.get("metadata_json", "{}"))
                except Exception:
                    doc["metadata"] = {}
                return doc
            return None

    def get_all_documents(self, sort_order: str = "DESC") -> List[Dict[str, Any]]:
        """Retrieves all documents sorted by upload date."""
        order = "ASC" if sort_order.upper() == "ASC" else "DESC"
        query = f"SELECT * FROM documents ORDER BY upload_date {order};"
        with self._connection() as conn:
            rows = conn.execute(query).fetchall()
            return [dict(r) for r in rows]

    def search_documents(
        self,
        search_query: Optional[str] = None,
        doc_type: Optional[str] = None,
        status: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        sort_order: str = "DESC",
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Multi-field search and filtering across:
        - original_filename
        - company
        - invoice_number
        - candidate_name
        - document_type
        - text_preview
        """
        conditions = []
        params = []

        if search_query and search_query.strip():
            term = f"%{search_query.strip()}%"
            conditions.append("""
                (original_filename LIKE ? OR
                 company LIKE ? OR
                 invoice_number LIKE ? OR
                 candidate_name LIKE ? OR
                 document_type LIKE ? OR
                 text_preview LIKE ?)
            """)
            params.extend([term, term, term, term, term, term])

        if doc_type and doc_type.lower() != "all":
            conditions.append("document_type = ?")
            params.append(doc_type)

        if status and status.lower() != "all":
            conditions.append("status = ?")
            params.append(status)

        if date_from:
            conditions.append("upload_date >= ?")
            params.append(date_from)

        if date_to:
            conditions.append("upload_date <= ?")
            params.append(date_to)

        where_clause = ""
        if conditions:
            where_clause = "WHERE " + " AND ".join(conditions)

        order = "ASC" if sort_order.upper() == "ASC" else "DESC"
        sql = f"""
            SELECT * FROM documents
            {where_clause}
            ORDER BY upload_date {order}
            LIMIT ? OFFSET ?;
        """
        params.extend([limit, offset])

        with self._connection() as conn:
            rows = conn.execute(sql, tuple(params)).fetchall()
            return [dict(r) for r in rows]

    def update_document_status(self, doc_id: int, new_status: str, reason: str = "") -> bool:
        """Updates the status and reason for a document."""
        query = "UPDATE documents SET status = ?, status_reason = ? WHERE id = ?;"
        with self._connection() as conn:
            cursor = conn.execute(query, (new_status, reason, doc_id))
            conn.commit()
            return cursor.rowcount > 0

    def delete_document(self, doc_id: int) -> Optional[Dict[str, Any]]:
        """
        Deletes a document from the database and returns its record
        so the caller can safely delete the stored physical file.
        """
        doc = self.get_document_by_id(doc_id)
        if not doc:
            return None

        query = "DELETE FROM documents WHERE id = ?;"
        with self._connection() as conn:
            conn.execute(query, (doc_id,))
            conn.commit()
        return doc

    def get_statistics(self) -> Dict[str, Any]:
        """Calculates authentic live KPI counters directly from SQLite."""
        with self._connection() as conn:
            total = conn.execute("SELECT COUNT(*) FROM documents;").fetchone()[0]
            invoices = conn.execute("SELECT COUNT(*) FROM documents WHERE document_type = 'Invoice';").fetchone()[0]
            resumes = conn.execute("SELECT COUNT(*) FROM documents WHERE document_type = 'Resume';").fetchone()[0]
            other = conn.execute("SELECT COUNT(*) FROM documents WHERE document_type NOT IN ('Invoice', 'Resume');").fetchone()[0]
            processed = conn.execute("SELECT COUNT(*) FROM documents WHERE status = 'Processed';").fetchone()[0]
            needs_review = conn.execute("SELECT COUNT(*) FROM documents WHERE status = 'Needs Review';").fetchone()[0]
            failed = conn.execute("SELECT COUNT(*) FROM documents WHERE status = 'Failed';").fetchone()[0]

            recent_rows = conn.execute(
                "SELECT * FROM documents ORDER BY upload_date DESC LIMIT 5;"
            ).fetchall()

            return {
                "total": total,
                "invoices": invoices,
                "resumes": resumes,
                "other": other,
                "processed": processed,
                "needs_review": needs_review,
                "failed": failed,
                "recent_documents": [dict(r) for r in recent_rows]
            }


_db_singleton: Optional[DatabaseManager] = None


def get_db(db_path: str = DEFAULT_DB_PATH) -> DatabaseManager:
    """Returns a singleton or instance of DatabaseManager."""
    global _db_singleton
    if _db_singleton is None or _db_singleton.db_path != db_path:
        _db_singleton = DatabaseManager(db_path)
    return _db_singleton
