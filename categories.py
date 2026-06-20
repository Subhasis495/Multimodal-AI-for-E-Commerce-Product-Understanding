"""
Default category taxonomy used for zero-shot CLIP classification.

CLIP doesn't need to be trained/fine-tuned on these labels — it scores the
product image against each label's text embedding and picks the closest
match. Replace or extend this list with your own product taxonomy.
"""

DEFAULT_CATEGORIES = [
    # Footwear
    "running shoes", "sneakers", "formal shoes", "sandals", "boots",
    # Apparel
    "t-shirt", "shirt", "jacket", "jeans", "dress", "saree",
    # Accessories
    "wrist watch", "handbag", "backpack", "sunglasses", "belt",
    # Electronics
    "smartphone", "laptop", "headphones", "smartwatch", "camera",
    # Home & furniture
    "sofa", "chair", "table lamp", "bookshelf", "bedsheet",
    # Kitchen
    "frying pan", "coffee mug", "water bottle", "blender",
    # Toys & sports
    "toy car", "board game", "bicycle", "skateboard", "cricket bat",
]
