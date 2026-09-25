"""
Document Classification Module
Implements TF-IDF feature extraction with multiple machine learning models:
1. Logistic Regression
2. Linear SVM
3. Multinomial Naive Bayes
4. Rule-Based Baseline
Includes honest confidence estimation and model persistence.
"""

import os
import joblib
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB

CLASS_INVOICE = "Invoice"
CLASS_RESUME = "Resume"
CLASS_OTHER = "Other"
TARGET_NAMES = [CLASS_INVOICE, CLASS_RESUME, CLASS_OTHER]

INVOICE_KEYWORDS = [
    "invoice", "invoice number", "invoice no", "inv-", "inv #", "inv no",
    "bill to", "billed to", "tax invoice", "amount due", "balance due",
    "total amount", "subtotal", "gstin", "payment terms", "due date",
    "item particulars", "description", "qty", "quantity", "unit price"
]

RESUME_KEYWORDS = [
    "resume", "curriculum vitae", "cv", "experience", "work experience",
    "education", "skills", "technical expertise", "projects", "summary",
    "professional summary", "employment", "certifications", "bachelor",
    "master", "university", "github", "linkedin", "competencies"
]


class RuleBasedClassifier:
    """Deterministic keyword scoring classifier as a baseline."""

    def predict(self, texts: List[str]) -> List[str]:
        predictions = []
        for text in texts:
            lower = text.lower()
            inv_score = sum(lower.count(kw) for kw in INVOICE_KEYWORDS)
            res_score = sum(lower.count(kw) for kw in RESUME_KEYWORDS)

            if inv_score > res_score and inv_score >= 2:
                predictions.append(CLASS_INVOICE)
            elif res_score > inv_score and res_score >= 2:
                predictions.append(CLASS_RESUME)
            else:
                predictions.append(CLASS_OTHER)
        return predictions

    def predict_single(self, text: str) -> Tuple[str, Dict[str, Any]]:
        lower = text.lower()
        inv_score = sum(lower.count(kw) for kw in INVOICE_KEYWORDS)
        res_score = sum(lower.count(kw) for kw in RESUME_KEYWORDS)

        if inv_score > res_score and inv_score >= 2:
            pred = CLASS_INVOICE
        elif res_score > inv_score and res_score >= 2:
            pred = CLASS_RESUME
        else:
            pred = CLASS_OTHER

        total = inv_score + res_score
        confidence_str = "Not Available"  # Rule-based lacks probabilistic calibration
        return pred, {
            "method": "Rule-Based Baseline",
            "invoice_score": inv_score,
            "resume_score": res_score,
            "confidence": confidence_str
        }


class DocumentClassifier:
    """Manages training, vectorization, and inference across ML classifiers."""

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=1500,
            sublinear_tf=True,
            stop_words="english"
        )
        self.models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "Linear SVM": LinearSVC(random_state=42, max_iter=2000),
            "Naive Bayes": MultinomialNB(alpha=0.1)
        }
        self.rule_based_baseline = RuleBasedClassifier()
        self.is_trained = False
        self.active_model_name = "Logistic Regression"

    def train(self, texts: List[str], labels: List[str]) -> None:
        """Fits vectorizer and trains all 3 ML models on labeled text data."""
        if not texts or not labels:
            raise ValueError("Training dataset cannot be empty.")

        X = self.vectorizer.fit_transform(texts)
        for name, model in self.models.items():
            model.fit(X, labels)

        self.is_trained = True

    def predict(
        self,
        text: str,
        model_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Predicts document category using the designated model.
        Returns predicted class, honest confidence percentage, and class probabilities.
        """
        selected_name = model_name or self.active_model_name

        # Route to Rule-based baseline if requested
        if selected_name == "Rule-Based Baseline":
            pred, details = self.rule_based_baseline.predict_single(text)
            return {
                "document_type": pred,
                "model_used": "Rule-Based Baseline",
                "confidence": details["confidence"],
                "confidence_val": None,
                "probabilities": None,
                "details": details
            }

        if not self.is_trained:
            # Fallback to rule-based if ML models have not been trained
            pred, details = self.rule_based_baseline.predict_single(text)
            return {
                "document_type": pred,
                "model_used": "Rule-Based Baseline (Model Unfitted Fallback)",
                "confidence": "Not Available",
                "confidence_val": None,
                "probabilities": None,
                "details": details
            }

        if selected_name not in self.models:
            selected_name = "Logistic Regression"

        model = self.models[selected_name]
        X_test = self.vectorizer.transform([text])
        prediction = model.predict(X_test)[0]

        # Extract calibrated probability if supported
        probabilities = {}
        confidence_str = "Not Available"
        confidence_val = None

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_test)[0]
            classes = model.classes_
            probabilities = {cls: round(float(p), 4) for cls, p in zip(classes, probs)}
            max_prob = float(np.max(probs))
            confidence_val = max_prob
            confidence_str = f"{round(max_prob * 100)}%"

        return {
            "document_type": prediction,
            "model_used": selected_name,
            "confidence": confidence_str,
            "confidence_val": confidence_val,
            "probabilities": probabilities,
            "details": {}
        }

    def save_model(self, filepath: str) -> None:
        """Serializes vectorizer and trained models to disk."""
        if not self.is_trained:
            raise RuntimeError("Cannot save an untrained classifier.")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        payload = {
            "vectorizer": self.vectorizer,
            "models": self.models,
            "active_model_name": self.active_model_name,
            "is_trained": self.is_trained
        }
        joblib.dump(payload, filepath)

    def load_model(self, filepath: str) -> bool:
        """Loads serialized models and vectorizer from disk."""
        if not os.path.isfile(filepath):
            return False
        try:
            payload = joblib.load(filepath)
            self.vectorizer = payload["vectorizer"]
            self.models = payload["models"]
            self.active_model_name = payload.get("active_model_name", "Logistic Regression")
            self.is_trained = payload.get("is_trained", True)
            return True
        except Exception:
            return False
