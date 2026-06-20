"""
Demo evaluation run for the homework's Section 5 (Evaluation).

This script feeds a small, hand-labeled sample set through the real
evaluate.py functions to produce a concrete report. The category
predictions/descriptions below are illustrative stand-ins for what
MultimodalProductPipeline.predict() would return — substitute them with
real pipeline.predict() outputs once you run app.py against your own
images (see README.md "Setup").

Run with:
    python demo_evaluation.py
"""

from evaluate import classify_accuracy, top_k_accuracy, simple_bleu1

# --- Sample 1: category predictions (10 labeled product images) ---
GROUND_TRUTH_CATEGORY = [
    "running shoes", "t-shirt", "laptop", "handbag", "smartwatch",
    "sofa", "frying pan", "bicycle", "sunglasses", "backpack",
]
TOP1_PREDICTION = [
    "running shoes", "t-shirt", "laptop", "backpack", "wrist watch",
    "sofa", "frying pan", "bicycle", "sunglasses", "handbag",
]
TOP3_PREDICTION = [
    ["running shoes", "sneakers", "boots"],
    ["t-shirt", "shirt", "jacket"],
    ["laptop", "smartphone", "headphones"],
    ["backpack", "handbag", "sunglasses"],
    ["wrist watch", "smartwatch", "smartphone"],
    ["sofa", "chair", "bookshelf"],
    ["frying pan", "coffee mug", "blender"],
    ["bicycle", "skateboard", "toy car"],
    ["sunglasses", "wrist watch", "handbag"],
    ["handbag", "sunglasses", "wrist watch"],  # true label "backpack" NOT in top-3 -> genuine miss
]

# --- Sample 2: generated vs. reference descriptions (5 items) ---
DESCRIPTION_PAIRS = [
    ("a pair of red running shoes on a white background",
     "red running shoes with white soles"),
    ("a black t-shirt hanging on a rack",
     "plain black cotton t-shirt"),
    ("a silver laptop open on a desk",
     "silver laptop computer with open lid"),
    ("a brown leather handbag with a gold buckle",
     "brown leather handbag with gold clasp"),
    ("a black smartwatch with a round face",
     "black smartwatch with round display"),
]


def main():
    print("=== Accuracy (category) ===")
    top1 = classify_accuracy(TOP1_PREDICTION, GROUND_TRUTH_CATEGORY)
    top3 = top_k_accuracy(TOP3_PREDICTION, GROUND_TRUTH_CATEGORY)
    print(f"Top-1 accuracy: {top1:.1%}  ({sum(p.lower()==g.lower() for p,g in zip(TOP1_PREDICTION, GROUND_TRUTH_CATEGORY))}/{len(GROUND_TRUTH_CATEGORY)} correct)")
    print(f"Top-3 accuracy: {top3:.1%}")

    print("\n=== Quality of descriptions (BLEU-1) ===")
    scores = [simple_bleu1(cand, ref) for cand, ref in DESCRIPTION_PAIRS]
    for (cand, ref), score in zip(DESCRIPTION_PAIRS, scores):
        print(f"  {score:.2f}  generated: \"{cand}\"")
        print(f"        reference: \"{ref}\"")
    print(f"Average BLEU-1: {sum(scores)/len(scores):.2f}")


if __name__ == "__main__":
    main()