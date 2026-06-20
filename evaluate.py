"""
Evaluation utilities matching the homework's Section 5 (Evaluation):
  - classify_accuracy / top_k_accuracy -> category prediction quality
  - simple_bleu1                       -> rough description quality check

These are intentionally lightweight (no extra dependencies). For a real
evaluation pipeline, swap simple_bleu1 for `sacrebleu` or `nltk`'s BLEU/ROUGE
implementations and run on a proper held-out labeled set.
"""

from typing import List


def classify_accuracy(predictions: List[str], ground_truth: List[str]) -> float:
    """Top-1 accuracy: fraction of predictions matching ground truth exactly."""
    if not predictions:
        return 0.0
    correct = sum(p.lower() == g.lower() for p, g in zip(predictions, ground_truth))
    return correct / len(predictions)


def top_k_accuracy(top_k_predictions: List[List[str]], ground_truth: List[str]) -> float:
    """Fraction of cases where the ground-truth category appears anywhere
    in that sample's top-k predicted list."""
    if not top_k_predictions:
        return 0.0
    hits = sum(
        g.lower() in [p.lower() for p in preds]
        for preds, g in zip(top_k_predictions, ground_truth)
    )
    return hits / len(top_k_predictions)


def simple_bleu1(candidate: str, reference: str) -> float:
    """Lightweight BLEU-1 (unigram precision) score for a quick sanity check
    of generated description quality against a reference description."""
    cand_tokens = candidate.lower().split()
    ref_tokens = set(reference.lower().split())
    if not cand_tokens:
        return 0.0
    matches = sum(1 for t in cand_tokens if t in ref_tokens)
    return matches / len(cand_tokens)
