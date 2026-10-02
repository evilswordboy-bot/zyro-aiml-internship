"""
Model Training Script
Trains all classifiers on data/train/ and serializes models to models/classifier.pkl.
"""

import os
import sys

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, base_dir)

from src.utils import generate_rich_sample_pdfs, load_dataset_from_dir
from src.classifier import DocumentClassifier
from src.evaluator import ModelEvaluator


def train_and_evaluate():
    print("1. Generating rich sample PDFs in samples/ ...")
    samples_dir = os.path.join(base_dir, "samples")
    generate_rich_sample_pdfs(samples_dir)

    print("2. Ingesting training dataset from data/train/ ...")
    train_dir = os.path.join(base_dir, "data", "train")
    train_texts, train_labels = load_dataset_from_dir(train_dir)
    print(f"   Loaded {len(train_texts)} training examples.")

    clf = DocumentClassifier()
    clf.train(train_texts, train_labels)

    model_path = os.path.join(base_dir, "models", "classifier.pkl")
    clf.save_model(model_path)
    print(f"3. Serialized trained model bundle to {model_path}")

    print("4. Evaluating models on unseen test dataset from data/test/ ...")
    test_dir = os.path.join(base_dir, "data", "test")
    test_texts, test_labels = load_dataset_from_dir(test_dir)
    print(f"   Loaded {len(test_texts)} test examples.")

    results = ModelEvaluator.evaluate_all_models(clf, test_texts, test_labels)

    print("\n" + "=" * 65)
    print(f"{'Model':<24} | {'Accuracy':<8} | {'Precision':<9} | {'Recall':<6} | {'F1-Score':<8}")
    print("-" * 65)
    for r in results["comparison_table"]:
        print(f"{r['Model']:<24} | {r['Accuracy']:<8.4f} | {r['Precision']:<9.4f} | {r['Recall']:<6.4f} | {r['F1-Score']:<8.4f}")
    print("=" * 65)

    print("\nSelected Best Model:", results["best_model_name"])
    return results


if __name__ == "__main__":
    train_and_evaluate()
