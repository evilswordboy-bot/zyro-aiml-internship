"""
Document Classification Engine for AI Document Intelligence Platform.
Implements:
1. Rule-based keyword matching with normalized confidence scores.
2. Scikit-learn TF-IDF + Logistic Regression ML classifier.
3. Hybrid classification mode with full explainability and feature attribution.
"""

import os
import re
import joblib
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .config import (
    DOC_TYPE_INVOICE,
    DOC_TYPE_RESUME,
    DOC_TYPE_OTHER,
    INVOICE_KEYWORDS,
    RESUME_KEYWORDS,
    MODELS_DIR,
)


@dataclass
class KeywordMatch:
    """Represents an individual keyword match with its weight and contribution."""
    keyword: str
    count: int
    weight: float
    total_contribution: float


@dataclass
class ClassificationResult:
    """Complete output of document classification including metrics and explanations."""
    document_type: str
    confidence: float                  # 0.0 to 1.0
    confidence_label: str              # High confidence / Medium confidence / Needs review
    method: str                        # Rule-Based / ML (TF-IDF + Logistic Regression) / Hybrid
    invoice_score: float
    resume_score: float
    top_keywords: List[KeywordMatch] = field(default_factory=list)
    top_ml_features: List[Tuple[str, float]] = field(default_factory=list)
    probabilities: Dict[str, float] = field(default_factory=dict)
    explanation: str = ""


# Synthetic training corpus representative of Invoices, Resumes, and Other documents
TRAINING_DOCUMENTS = [
    # Invoices
    ("Tax Invoice Invoice Number INV-1024 Date 02-09-2026 Bill To ABC Technologies Description Web Development Subtotal PKR 125,000 Total Amount Due Payment Terms Net 30", DOC_TYPE_INVOICE),
    ("Commercial Invoice No 98432 Billed to Nexus Corp Item Cloud Hosting Quantity 5 Unit Price $250 Balance Due $1,250 Remit to Nexus Solutions", DOC_TYPE_INVOICE),
    ("INVOICE # 5501 Date October 12 2025 Vendor Acme Supplies Total PKR 45,000 Tax Invoice GST Included Remit payment within 14 days", DOC_TYPE_INVOICE),
    ("Invoice 2026-004 Bill to Global Enterprise Amount Due $8,400.00 Subtotal $8,000 Tax $400 Purchase Order PO-9921 Payment due upon receipt", DOC_TYPE_INVOICE),
    ("Customer Invoice No: 7721 Date: 2026-01-15 Bill To: Apex Data Systems Items: Consulting Hours Rate: $150 Total Amount: $4,500 Paid: $0 Balance Due: $4,500", DOC_TYPE_INVOICE),
    ("Proforma Invoice Ref: PI-3021 Buyer: Horizon Tech Subtotal EUR 12,000 VAT 20% EUR 2,400 Grand Total EUR 14,400 Bank transfer remittance details below", DOC_TYPE_INVOICE),
    ("Tax Invoice Inv No: 8812 Date: 2026-03-01 Billed to: Apex Technologies Services: Software Engineering Total Amount PKR 250,000 Due Date: March 31 2026", DOC_TYPE_INVOICE),
    ("Monthly Invoice Number: INV-6002 Client: Stellar Soft Total Amount Due: $3,200.00 Payment terms: Net 15 days Thank you for your business", DOC_TYPE_INVOICE),
    
    # Resumes
    ("John Doe Curriculum Vitae Email john.doe@example.com Phone +1-555-0199 Experience Senior Python Developer Education Bachelor of Science in Computer Science Skills Python PyTorch Scikit-Learn Docker Kubernetes FastAPI PostgreSQL", DOC_TYPE_RESUME),
    ("Resume Sarah Jenkins Software Engineer Professional Experience 5 years in Full Stack Development React TypeScript Node.js Education University of Toronto BS Software Engineering Projects Microservices architecture", DOC_TYPE_RESUME),
    ("Alex Morgan AI ML Engineer alex.morgan@email.com Technical Skills Deep Learning NLP TensorFlow PyTorch Pandas NumPy Work Experience Machine Learning Researcher at AI Labs Certifications AWS Certified Machine Learning Specialist", DOC_TYPE_RESUME),
    ("Curriculum Vitae David Kim Data Scientist Work Experience Built predictive forecasting models with LightGBM and Scikit-Learn Education Master of Science in Data Science University of Washington Skills Python SQL Spark Tableau", DOC_TYPE_RESUME),
    ("Resume Priya Sharma Mobile Application Developer Experience Android Studio Kotlin Swift Flutter UI UX Design Education Bachelor of Engineering in Information Technology Achievements Published 4 apps on Google Play Store", DOC_TYPE_RESUME),
    ("Profile & Resume Marcus Vance DevOps Engineer Professional Experience CI/CD pipelines Terraform Docker Kubernetes AWS Azure Education BS Computer Information Systems Skills Linux Bash Python Prometheus Grafana", DOC_TYPE_RESUME),
    ("Emily Zhang Curriculum Vitae Experience Junior AI Engineer Intern at Apex Tech Education BS Artificial Intelligence Skills Python OpenCV Scikit-Learn Hugging Face Git", DOC_TYPE_RESUME),
    ("Resume Michael Brown Cloud Architect Experience 8 years in Enterprise Cloud Migration AWS Azure GCP Docker Kubernetes Microservices Education BS Computer Science", DOC_TYPE_RESUME),
    
    # Other / General documents
    ("Meeting Minutes Project Lifeline Discussion regarding upcoming feature release timeline and team bandwidth. Action items assigned to Sarah and Tom. Next sprint planning on Monday.", DOC_TYPE_OTHER),
    ("Research Paper Abstract: Deep Convolutional Neural Networks for Automated Solar Flare Forecasting. Solar flares represent explosive coronal releases of energy. We evaluate temporal attention mechanisms.", DOC_TYPE_OTHER),
    ("Terms of Service and Privacy Policy. By accessing this web application you agree to be bound by these Terms. All user data is processed in accordance with national privacy laws.", DOC_TYPE_OTHER),
    ("Software Architecture Design Document: The document intelligence microservice processes incoming multipart payload through queue-based pub sub messaging. System latency SLA is 200ms.", DOC_TYPE_OTHER),
    ("Employee Handbook and Company Policy: Working hours standard leave entitlements remote work etiquette and expense reimbursement protocols for all staff members.", DOC_TYPE_OTHER),
    ("Quarterly Financial Report Q3: Operating revenues increased by 14% year over year driven by enterprise subscription growth. Capital expenditures remained within projected budget.", DOC_TYPE_OTHER),
    ("Technical Blog Post: Getting Started with Docker and Containerization for Beginners. Containers package your code and all its dependencies so the application runs quickly and reliably.", DOC_TYPE_OTHER),
    ("Weekly Progress Status Report: Sprint 24 completion rate stands at 87%. Resolved 14 blocker bugs. Ready for QA staging deployment.", DOC_TYPE_OTHER),
]


class RuleBasedClassifier:
    """Classifies documents using weighted keyword dictionaries with normalized scores."""

    def __init__(self, min_score_threshold: float = 3.0):
        self.min_score_threshold = min_score_threshold

    def classify(self, text: str) -> ClassificationResult:
        lower_text = text.lower()
        inv_matches: List[KeywordMatch] = []
        res_matches: List[KeywordMatch] = []

        total_inv_score = 0.0
        for kw, weight in INVOICE_KEYWORDS.items():
            pattern = rf"\b{re.escape(kw)}\b"
            matches = len(re.findall(pattern, lower_text))
            if matches > 0:
                contribution = matches * weight
                total_inv_score += contribution
                inv_matches.append(KeywordMatch(kw, matches, weight, contribution))

        total_res_score = 0.0
        for kw, weight in RESUME_KEYWORDS.items():
            pattern = rf"\b{re.escape(kw)}\b"
            matches = len(re.findall(pattern, lower_text))
            if matches > 0:
                contribution = matches * weight
                total_res_score += contribution
                res_matches.append(KeywordMatch(kw, matches, weight, contribution))

        # Sort matches by total contribution descending
        inv_matches.sort(key=lambda x: x.total_contribution, reverse=True)
        res_matches.sort(key=lambda x: x.total_contribution, reverse=True)

        # Decision Logic
        max_score = max(total_inv_score, total_res_score)
        if max_score < self.min_score_threshold:
            doc_type = DOC_TYPE_OTHER
            confidence = 0.85 if len(text.strip()) > 50 else 0.50
            explanation = "Document did not contain significant invoice or resume keywords; classified as Other."
            top_keywords = []
        elif total_inv_score > total_res_score:
            doc_type = DOC_TYPE_INVOICE
            ratio = total_inv_score / (total_inv_score + total_res_score + 1e-5)
            confidence = min(0.99, max(0.55, ratio))
            top_kw_names = [m.keyword for m in inv_matches[:3]]
            explanation = f"Classified as Invoice with high signal from keywords: {', '.join(top_kw_names)}."
            top_keywords = inv_matches[:5]
        else:
            doc_type = DOC_TYPE_RESUME
            ratio = total_res_score / (total_inv_score + total_res_score + 1e-5)
            confidence = min(0.99, max(0.55, ratio))
            top_kw_names = [m.keyword for m in res_matches[:3]]
            explanation = f"Classified as Resume with strong indicators: {', '.join(top_kw_names)}."
            top_keywords = res_matches[:5]

        # Determine confidence label
        if confidence >= 0.80:
            confidence_label = "High confidence"
        elif confidence >= 0.60:
            confidence_label = "Medium confidence"
        else:
            confidence_label = "Needs review"

        total_combined = total_inv_score + total_res_score + 1e-5
        probabilities = {
            DOC_TYPE_INVOICE: round(total_inv_score / total_combined, 4) if doc_type != DOC_TYPE_OTHER else 0.05,
            DOC_TYPE_RESUME: round(total_res_score / total_combined, 4) if doc_type != DOC_TYPE_OTHER else 0.05,
            DOC_TYPE_OTHER: 0.90 if doc_type == DOC_TYPE_OTHER else round(1.0 - (max_score / total_combined), 4)
        }

        return ClassificationResult(
            document_type=doc_type,
            confidence=round(confidence, 4),
            confidence_label=confidence_label,
            method="Rule-Based Keyword Scoring",
            invoice_score=round(total_inv_score, 2),
            resume_score=round(total_res_score, 2),
            top_keywords=top_keywords,
            probabilities=probabilities,
            explanation=explanation
        )


class MLDocumentClassifier:
    """Scikit-Learn TF-IDF + Logistic Regression document classifier."""

    def __init__(self):
        self.model_path = MODELS_DIR / "doc_classifier_pipeline.joblib"
        self.pipeline: Optional[Pipeline] = None
        self._initialize_or_load_model()

    def _initialize_or_load_model(self):
        """Loads serialized model if present, otherwise trains on domain corpus."""
        if self.model_path.exists():
            try:
                self.pipeline = joblib.load(self.model_path)
                return
            except Exception:
                pass

        # Train new pipeline
        texts = [doc[0] for doc in TRAINING_DOCUMENTS]
        labels = [doc[1] for doc in TRAINING_DOCUMENTS]

        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
            ("clf", LogisticRegression(C=2.0, max_iter=300, random_state=42))
        ])
        self.pipeline.fit(texts, labels)

        # Save model
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        try:
            joblib.dump(self.pipeline, self.model_path)
        except Exception:
            pass

    def classify(self, text: str) -> ClassificationResult:
        if not text or not text.strip():
            return ClassificationResult(
                document_type=DOC_TYPE_OTHER,
                confidence=0.5,
                confidence_label="Needs review",
                method="TF-IDF + Logistic Regression",
                invoice_score=0.0,
                resume_score=0.0,
                explanation="Empty text cannot be reliably classified."
            )

        probas = self.pipeline.predict_proba([text])[0]
        classes = self.pipeline.classes_
        proba_dict = {cls: float(round(p, 4)) for cls, p in zip(classes, probas)}

        best_idx = int(np.argmax(probas))
        predicted_class = classes[best_idx]
        confidence = float(probas[best_idx])

        # Feature importance / attribution
        vectorizer: TfidfVectorizer = self.pipeline.named_steps["tfidf"]
        classifier: LogisticRegression = self.pipeline.named_steps["clf"]

        tfidf_vec = vectorizer.transform([text])
        feature_names = vectorizer.get_feature_names_out()

        # Class index in classifier
        class_idx = list(classifier.classes_).index(predicted_class)
        coefficients = classifier.coef_[class_idx]

        # Calculate word attribution = tfidf_value * coef
        word_indices = tfidf_vec.nonzero()[1]
        feature_scores = []
        for idx in word_indices:
            score = tfidf_vec[0, idx] * coefficients[idx]
            if score > 0:
                feature_scores.append((feature_names[idx], float(round(score, 4))))

        feature_scores.sort(key=lambda x: x[1], reverse=True)
        top_features = feature_scores[:5]

        if confidence >= 0.80:
            conf_label = "High confidence"
        elif confidence >= 0.60:
            conf_label = "Medium confidence"
        else:
            conf_label = "Needs review"

        top_words_str = ", ".join([f"'{f[0]}'" for f in top_features[:3]]) if top_features else "general vocabulary"
        explanation = f"ML model predicted '{predicted_class}' with {confidence*100:.1f}% probability driven by features: {top_words_str}."

        return ClassificationResult(
            document_type=predicted_class,
            confidence=round(confidence, 4),
            confidence_label=conf_label,
            method="TF-IDF + Logistic Regression (ML)",
            invoice_score=proba_dict.get(DOC_TYPE_INVOICE, 0.0),
            resume_score=proba_dict.get(DOC_TYPE_RESUME, 0.0),
            top_ml_features=top_features,
            probabilities=proba_dict,
            explanation=explanation
        )


class HybridClassifier:
    """Combines Rule-Based keyword heuristics and Scikit-Learn ML predictions."""

    def __init__(self):
        self.rule_engine = RuleBasedClassifier()
        self.ml_engine = MLDocumentClassifier()

    def classify(self, text: str, mode: str = "Hybrid") -> ClassificationResult:
        """
        Classifies document text using the specified mode.
        Modes: 'Hybrid', 'Rule-Based', 'ML'
        """
        if mode == "Rule-Based":
            return self.rule_engine.classify(text)
        elif mode == "ML":
            return self.ml_engine.classify(text)

        # Hybrid Ensemble
        rule_res = self.rule_engine.classify(text)
        ml_res = self.ml_engine.classify(text)

        # Agreement check
        if rule_res.document_type == ml_res.document_type:
            doc_type = rule_res.document_type
            combined_conf = min(0.99, max(rule_res.confidence, ml_res.confidence) + 0.05)
            explanation = (
                f"Consensus reached between Rule-Based heuristics and ML Classifier. "
                f"Identified as {doc_type} with strong signal."
            )
        else:
            # When rule engine found strong explicit keywords (> 6.0 score), prioritize rule
            if rule_res.document_type != DOC_TYPE_OTHER and max(rule_res.invoice_score, rule_res.resume_score) >= 6.0:
                doc_type = rule_res.document_type
                combined_conf = rule_res.confidence
                explanation = f"Rule-based keyword scoring took precedence due to strong explicit term matches ({rule_res.explanation})."
            else:
                doc_type = ml_res.document_type
                combined_conf = ml_res.confidence
                explanation = f"Machine learning classifier took precedence ({ml_res.explanation})."

        if combined_conf >= 0.80:
            conf_label = "High confidence"
        elif combined_conf >= 0.60:
            conf_label = "Medium confidence"
        else:
            conf_label = "Needs review"

        return ClassificationResult(
            document_type=doc_type,
            confidence=round(combined_conf, 4),
            confidence_label=conf_label,
            method="Hybrid (Rules + ML Ensemble)",
            invoice_score=rule_res.invoice_score,
            resume_score=rule_res.resume_score,
            top_keywords=rule_res.top_keywords,
            top_ml_features=ml_res.top_ml_features,
            probabilities=ml_res.probabilities,
            explanation=explanation
        )
