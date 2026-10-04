"""
RAG Engine & Grounded Document Intelligence Assistant.
Implements semantic chunking, TF-IDF vector retrieval, multi-document grounding,
and strict anti-hallucination guardrails.
Week 6: Final AI Assistant & RAG Validation (Section 7, Tests 13 & 14).
"""

import re
import os
from typing import List, Dict, Any, Optional, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

NO_INFO_RESPONSE = "The available documents do not contain sufficient information to answer this question."


class DocumentChunk:
    """Represents a single semantically indexed chunk of a document."""
    def __init__(self, doc_id: int, filename: str, doc_type: str, chunk_id: int, text: str):
        self.doc_id = doc_id
        self.filename = filename
        self.doc_type = doc_type
        self.chunk_id = chunk_id
        self.text = text

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "filename": self.filename,
            "doc_type": self.doc_type,
            "chunk_id": self.chunk_id,
            "text": self.text
        }


class RAGEngine:
    """
    Retrieval-Augmented Generation Engine with offline vector indexing
    and strict grounded question answering.
    """

    def __init__(
        self,
        chunk_size_words: int = 150,
        chunk_overlap_words: int = 30,
        similarity_threshold: float = 0.12,
        top_k: int = 3
    ):
        self.chunk_size_words = chunk_size_words
        self.chunk_overlap_words = chunk_overlap_words
        self.similarity_threshold = similarity_threshold
        self.top_k = top_k

        self.chunks: List[DocumentChunk] = []
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.chunk_vectors: Optional[Any] = None
        self.is_indexed: bool = False

    def chunk_document(
        self,
        doc_id: int,
        filename: str,
        doc_type: str,
        text: str
    ) -> List[DocumentChunk]:
        """
        Splits raw document text into overlapping semantic word chunks.
        """
        if not text or not text.strip():
            return []

        words = text.split()
        if not words:
            return []

        doc_chunks: List[DocumentChunk] = []
        step = max(1, self.chunk_size_words - self.chunk_overlap_words)
        chunk_idx = 0

        for i in range(0, len(words), step):
            chunk_words = words[i:i + self.chunk_size_words]
            chunk_text = " ".join(chunk_words).strip()
            if len(chunk_text) >= 15:  # ignore trivial fragments
                doc_chunks.append(
                    DocumentChunk(
                        doc_id=doc_id,
                        filename=filename,
                        doc_type=doc_type,
                        chunk_id=chunk_idx,
                        text=chunk_text
                    )
                )
                chunk_idx += 1

        return doc_chunks

    def build_index(self, documents: List[Dict[str, Any]]) -> int:
        """
        Builds the vector space index over all provided documents.
        Returns total number of chunks indexed.
        """
        self.chunks = []
        for doc in documents:
            doc_id = doc.get("id", 0)
            filename = doc.get("original_filename", "untitled")
            doc_type = doc.get("document_type", "Other")
            # Prefer raw_text if present, fallback to text_preview
            full_text = doc.get("raw_text") or doc.get("text_preview") or ""
            doc_chunks = self.chunk_document(doc_id, filename, doc_type, full_text)
            self.chunks.extend(doc_chunks)

        if not self.chunks:
            self.vectorizer = None
            self.chunk_vectors = None
            self.is_indexed = False
            return 0

        corpus = [c.text for c in self.chunks]
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words="english",
            max_features=12000,
            sublinear_tf=True
        )
        self.chunk_vectors = self.vectorizer.fit_transform(corpus)
        self.is_indexed = True
        return len(self.chunks)

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Tuple[DocumentChunk, float]]:
        """
        Retrieves top_k most relevant chunks for a query using cosine similarity.
        """
        k = top_k or self.top_k
        if not self.is_indexed or not self.vectorizer or not self.chunks:
            return []

        query_clean = query.strip()
        if not query_clean:
            return []

        query_vec = self.vectorizer.transform([query_clean])
        sims = cosine_similarity(query_vec, self.chunk_vectors).flatten()

        # Get top-k indices sorted descending
        top_indices = np.argsort(sims)[::-1][:k]

        results = []
        for idx in top_indices:
            score = float(sims[idx])
            results.append((self.chunks[idx], score))

        return results

    def answer_question(
        self,
        query: str,
        top_k: Optional[int] = None,
        custom_threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Executes grounded question answering with anti-hallucination guarantee.
        If no relevant chunks exist above threshold, strictly returns NO_INFO_RESPONSE.
        """
        threshold = custom_threshold if custom_threshold is not None else self.similarity_threshold
        k = top_k or self.top_k

        if not self.is_indexed or not self.chunks:
            return {
                "answered": False,
                "answer": NO_INFO_RESPONSE,
                "confidence": 0.0,
                "sources": [],
                "status": "NO_DOCUMENTS_INDEXED",
                "query": query
            }

        retrieved = self.retrieve(query, top_k=k)
        # Filter chunks that meet the relevance threshold
        filtered = [(chunk, score) for chunk, score in retrieved if score >= threshold]

        # Guardrail: Insufficient Information check (Section 6 Test 14)
        if not filtered:
            return {
                "answered": False,
                "answer": NO_INFO_RESPONSE,
                "confidence": round(float(retrieved[0][1]), 3) if retrieved else 0.0,
                "sources": [],
                "status": "INSUFFICIENT_INFORMATION",
                "query": query
            }

        # Build grounded response with source citations
        best_chunk, top_score = filtered[0]
        sources = []
        context_blocks = []

        for chunk, score in filtered:
            sources.append({
                "doc_id": chunk.doc_id,
                "filename": chunk.filename,
                "doc_type": chunk.doc_type,
                "chunk_id": chunk.chunk_id,
                "relevance_pct": round(score * 100, 1),
                "text_snippet": chunk.text[:180] + ("..." if len(chunk.text) > 180 else "")
            })
            context_blocks.append(f"[{chunk.filename}]: {chunk.text}")

        # Deterministic Grounded Synthesis (100% Offline)
        grounded_answer = self._synthesize_grounded_answer(query, filtered)

        return {
            "answered": True,
            "answer": grounded_answer,
            "confidence": round(top_score, 3),
            "sources": sources,
            "status": "GROUNDED_ANSWER",
            "query": query
        }

    def _synthesize_grounded_answer(
        self,
        query: str,
        ranked_chunks: List[Tuple[DocumentChunk, float]]
    ) -> str:
        """
        Extracts and synthesizes answering facts strictly from retrieved chunks.
        """
        query_words = set(re.findall(r"\w+", query.lower()))
        # Remove common query question words
        stop_q = {"what", "who", "where", "when", "how", "is", "are", "the", "a", "an", "does", "do", "in", "of", "for", "much", "total", "amount", "phone", "email", "skills"}
        informative_q_words = query_words - stop_q

        extracted_sentences = []
        seen_sentences = set()

        for chunk, score in ranked_chunks:
            # Split chunk into sentences
            sentences = re.split(r"(?<=[.!?\n])\s+", chunk.text)
            for s in sentences:
                s_clean = s.strip()
                if not s_clean or len(s_clean) < 10 or s_clean in seen_sentences:
                    continue

                s_words = set(re.findall(r"\w+", s_clean.lower()))
                overlap = len(s_words & informative_q_words)
                # Specific entity questions
                has_entity_signal = False
                if any(k in query.lower() for k in ["total", "amount", "price", "cost", "pay"]) and re.search(r"[$€£₹]|\b\d+\.\d{2}\b", s_clean):
                    has_entity_signal = True
                elif any(k in query.lower() for k in ["email", "contact", "phone", "number"]) and re.search(r"@|\b\d{3}[-.]?\d{3}\b", s_clean):
                    has_entity_signal = True
                elif any(k in query.lower() for k in ["skill", "technolog", "experience"]) and re.search(r"\b(Python|Machine Learning|SQL|Cloud|TensorFlow|PyTorch|React)\b", s_clean, re.I):
                    has_entity_signal = True

                if overlap > 0 or has_entity_signal:
                    extracted_sentences.append((s_clean, chunk.filename, overlap + (2 if has_entity_signal else 0)))
                    seen_sentences.add(s_clean)

        # Sort extracted sentences by relevance
        extracted_sentences.sort(key=lambda x: x[2], reverse=True)

        if extracted_sentences:
            top_facts = extracted_sentences[:3]
            answer_parts = []
            for s, fn, _ in top_facts:
                answer_parts.append(f"• {s} *(Source: {fn})*")
            return "Based on the retrieved documents:\n\n" + "\n".join(answer_parts)

        # Fallback to top chunk snippet directly with source attribution
        primary_chunk, _ = ranked_chunks[0]
        return f"Based on {primary_chunk.filename}:\n\n> \"{primary_chunk.text[:280]}...\""
