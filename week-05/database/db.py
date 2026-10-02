"""
SQLite Database Layer for AI Document Intelligence & Workflow Platform.
Provides robust schema management, migrations, parameterized queries,
full CRUD operations, and complete audit logging (Week 5).
"""

import os
import sqlite3
import json
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from contextlib import contextmanager

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DB_PATH = os.path.join(BASE_DIR, "data", "documents.db")


class DatabaseManager:
    """Manages SQLite connection lifecycle, migrations, CRUD, and audit logging."""

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
        """Initializes the database schema, handles non-destructive migrations, and indices."""
        with self._connection() as conn:
            # 1. Base documents table
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

            # 2. Week 5 Schema Migration: Add new workflow columns if not already present
            existing_cols = {
                row["name"] for row in conn.execute("PRAGMA table_info(documents);").fetchall()
            }

            new_columns = [
                ("current_status", "TEXT"),
                ("predicted_type", "TEXT"),
                ("confidence_score", "REAL"),
                ("validation_result_json", "TEXT DEFAULT '{}'"),
                ("missing_fields_json", "TEXT DEFAULT '[]'"),
                ("invalid_fields_json", "TEXT DEFAULT '[]'"),
                ("created_at", "TEXT"),
                ("updated_at", "TEXT"),
            ]

            for col_name, col_type in new_columns:
                if col_name not in existing_cols:
                    conn.execute(f"ALTER TABLE documents ADD COLUMN {col_name} {col_type};")

            # Backfill existing rows if needed
            conn.execute("""
                UPDATE documents
                SET current_status = status
                WHERE current_status IS NULL;
            """)
            conn.execute("""
                UPDATE documents
                SET predicted_type = document_type
                WHERE predicted_type IS NULL;
            """)
            conn.execute("""
                UPDATE documents
                SET created_at = upload_date, updated_at = upload_date
                WHERE created_at IS NULL;
            """)

            # 3. Week 5 Audit Logs Table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_logs (
                    audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id INTEGER NOT NULL,
                    action TEXT NOT NULL,
                    previous_status TEXT,
                    new_status TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    reason TEXT DEFAULT '',
                    reviewer_note TEXT DEFAULT '',
                    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
                );
            """)

            # 4. Performance Indices
            conn.execute("CREATE INDEX IF NOT EXISTS idx_doc_hash ON documents(file_hash);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_doc_type ON documents(document_type);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_doc_status ON documents(status);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_doc_curr_status ON documents(current_status);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_doc_upload_date ON documents(upload_date);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_doc_company ON documents(company);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_doc_inv_num ON documents(invoice_number);")

            conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_doc_id ON audit_logs(document_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_action ON audit_logs(action);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_logs(timestamp);")

            conn.commit()

    def add_document(self, doc_data: Dict[str, Any]) -> int:
        """
        Inserts a newly processed document record.
        Uses parameterized queries to prevent SQL injection.
        """
        now_iso = datetime.now().isoformat()
        upload_date = doc_data.get("upload_date") or now_iso
        status = doc_data.get("status") or doc_data.get("current_status", "Completed")
        current_status = doc_data.get("current_status") or status
        predicted_type = doc_data.get("predicted_type") or doc_data.get("document_type", "Other")

        metadata_json = doc_data.get("metadata_json")
        if isinstance(metadata_json, dict):
            metadata_json = json.dumps(metadata_json)
        elif not metadata_json:
            metadata_json = "{}"

        val_res = doc_data.get("validation_result_json", "{}")
        if isinstance(val_res, dict):
            val_res = json.dumps(val_res)

        missing_f = doc_data.get("missing_fields_json", "[]")
        if isinstance(missing_f, list):
            missing_f = json.dumps(missing_f)

        invalid_f = doc_data.get("invalid_fields_json", "[]")
        if isinstance(invalid_f, list):
            invalid_f = json.dumps(invalid_f)

        query = """
            INSERT INTO documents (
                original_filename, stored_filename, document_type, upload_date,
                company, invoice_number, total_amount, candidate_name,
                candidate_email, candidate_phone, candidate_skills,
                file_path, text_preview, file_hash, status, status_reason,
                metadata_json, current_status, predicted_type, confidence_score,
                validation_result_json, missing_fields_json, invalid_fields_json,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
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
            status,
            doc_data.get("status_reason", ""),
            metadata_json,
            current_status,
            predicted_type,
            doc_data.get("confidence_score"),
            val_res,
            missing_f,
            invalid_f,
            doc_data.get("created_at", upload_date),
            doc_data.get("updated_at", upload_date)
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
                try:
                    doc["validation_result"] = json.loads(doc.get("validation_result_json", "{}"))
                except Exception:
                    doc["validation_result"] = {}
                try:
                    doc["missing_fields"] = json.loads(doc.get("missing_fields_json", "[]"))
                except Exception:
                    doc["missing_fields"] = []
                try:
                    doc["invalid_fields"] = json.loads(doc.get("invalid_fields_json", "[]"))
                except Exception:
                    doc["invalid_fields"] = []
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
        - current_status
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
            # Match either status or current_status, handling Completed/Processed equivalence
            if status in ("Completed", "Processed"):
                conditions.append("(status IN ('Completed', 'Processed') OR current_status IN ('Completed', 'Processed'))")
            else:
                conditions.append("(status = ? OR current_status = ?)")
                params.extend([status, status])

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

    def update_document_status(
        self,
        doc_id: int,
        new_status: str,
        reason: str = ""
    ) -> bool:
        """Updates the status, current_status, status_reason, and updated_at for a document."""
        now_iso = datetime.now().isoformat()
        query = """
            UPDATE documents
            SET status = ?, current_status = ?, status_reason = ?, updated_at = ?
            WHERE id = ?;
        """
        with self._connection() as conn:
            cursor = conn.execute(query, (new_status, new_status, reason, now_iso, doc_id))
            conn.commit()
            return cursor.rowcount > 0

    def delete_document(self, doc_id: int) -> Optional[Dict[str, Any]]:
        """
        Deletes a document from the database and returns its record
        so the caller can safely delete the stored physical file.
        Associated audit logs cascade or are cleaned up.
        """
        doc = self.get_document_by_id(doc_id)
        if not doc:
            return None

        with self._connection() as conn:
            conn.execute("DELETE FROM audit_logs WHERE document_id = ?;", (doc_id,))
            conn.execute("DELETE FROM documents WHERE id = ?;", (doc_id,))
            conn.commit()
        return doc

    # -------------------------------------------------------------------------
    # Week 5: Audit Logging Operations
    # -------------------------------------------------------------------------

    def log_audit_event(
        self,
        document_id: int,
        action: str,
        previous_status: Optional[str],
        new_status: str,
        reason: str = "",
        reviewer_note: str = ""
    ) -> int:
        """
        Appends an immutable audit event for a document's workflow transition.
        """
        now_iso = datetime.now().isoformat()
        query = """
            INSERT INTO audit_logs (
                document_id, action, previous_status, new_status,
                timestamp, reason, reviewer_note
            ) VALUES (?, ?, ?, ?, ?, ?, ?);
        """
        params = (
            document_id,
            action,
            previous_status or "None",
            new_status,
            now_iso,
            reason or "",
            reviewer_note or ""
        )
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            conn.commit()
            return cursor.lastrowid

    def get_document_audit_history(self, document_id: int) -> List[Dict[str, Any]]:
        """Retrieves chronological audit trail for a specific document."""
        query = """
            SELECT * FROM audit_logs
            WHERE document_id = ?
            ORDER BY audit_id ASC;
        """
        with self._connection() as conn:
            rows = conn.execute(query, (document_id,)).fetchall()
            return [dict(r) for r in rows]

    def get_recent_audit_events(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Retrieves the global audit activity feed with document metadata joined.
        """
        query = """
            SELECT
                a.audit_id,
                a.document_id,
                a.action,
                a.previous_status,
                a.new_status,
                a.timestamp,
                a.reason,
                a.reviewer_note,
                d.original_filename,
                d.document_type
            FROM audit_logs a
            LEFT JOIN documents d ON a.document_id = d.id
            ORDER BY a.audit_id DESC
            LIMIT ?;
        """
        with self._connection() as conn:
            rows = conn.execute(query, (limit,)).fetchall()
            return [dict(r) for r in rows]

    def get_documents_by_status(self, status: str) -> List[Dict[str, Any]]:
        """Retrieves all documents matching a specific status."""
        query = """
            SELECT * FROM documents
            WHERE status = ? OR current_status = ?
            ORDER BY upload_date DESC;
        """
        with self._connection() as conn:
            rows = conn.execute(query, (status, status)).fetchall()
            return [dict(r) for r in rows]

    def get_statistics(self) -> Dict[str, Any]:
        """Calculates authentic live KPI counters directly from SQLite."""
        with self._connection() as conn:
            total = conn.execute("SELECT COUNT(*) FROM documents;").fetchone()[0]
            invoices = conn.execute("SELECT COUNT(*) FROM documents WHERE document_type = 'Invoice';").fetchone()[0]
            resumes = conn.execute("SELECT COUNT(*) FROM documents WHERE document_type = 'Resume';").fetchone()[0]
            other = conn.execute("SELECT COUNT(*) FROM documents WHERE document_type NOT IN ('Invoice', 'Resume');").fetchone()[0]

            completed = conn.execute("""
                SELECT COUNT(*) FROM documents
                WHERE status IN ('Completed', 'Processed') OR current_status IN ('Completed', 'Processed');
            """).fetchone()[0]

            needs_review = conn.execute("""
                SELECT COUNT(*) FROM documents
                WHERE status = 'Needs Review' OR current_status = 'Needs Review';
            """).fetchone()[0]

            approved = conn.execute("""
                SELECT COUNT(*) FROM documents
                WHERE status = 'Approved' OR current_status = 'Approved';
            """).fetchone()[0]

            rejected = conn.execute("""
                SELECT COUNT(*) FROM documents
                WHERE status = 'Rejected' OR current_status = 'Rejected';
            """).fetchone()[0]

            failed = conn.execute("""
                SELECT COUNT(*) FROM documents
                WHERE status = 'Failed' OR current_status = 'Failed';
            """).fetchone()[0]

            total_audit_events = conn.execute("SELECT COUNT(*) FROM audit_logs;").fetchone()[0]

            recent_rows = conn.execute(
                "SELECT * FROM documents ORDER BY upload_date DESC LIMIT 5;"
            ).fetchall()

            return {
                "total": total,
                "invoices": invoices,
                "resumes": resumes,
                "other": other,
                "completed": completed,
                "processed": completed,  # alias for backward compat
                "needs_review": needs_review,
                "approved": approved,
                "rejected": rejected,
                "failed": failed,
                "total_audit_events": total_audit_events,
                "recent_documents": [dict(r) for r in recent_rows]
            }


_db_singleton: Optional[DatabaseManager] = None


def get_db(db_path: str = DEFAULT_DB_PATH) -> DatabaseManager:
    """Returns a singleton or instance of DatabaseManager."""
    global _db_singleton
    if _db_singleton is None or _db_singleton.db_path != db_path:
        _db_singleton = DatabaseManager(db_path)
    return _db_singleton
