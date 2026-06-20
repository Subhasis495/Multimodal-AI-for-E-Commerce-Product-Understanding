"""
Core multimodal pipeline for the product understanding system.

Maps directly onto the homework's system design:
    Step 2 (preprocessing)        -> MultimodalProductPipeline.predict() / processors
    Step 3 (multimodal model)     -> CLIP + BLIP backbones
    Step 4a (classification head) -> MultimodalProductPipeline.classify()  [CLIP, zero-shot]
    Step 4b (caption generator)   -> MultimodalProductPipeline.describe() [BLIP]

CLIP and BLIP are used together because they're complementary:
  - CLIP shares one embedding space for images and text, which makes it a
    natural fit for zero-shot category matching (no fine-tuning needed —
    just compare the image against candidate category text prompts).
  - BLIP is trained specifically to generate free-form captions, which CLIP
    cannot do on its own.
"""

from dataclasses import dataclass
from typing import List, Tuple

import torch
from PIL import Image
from transformers import (
    BlipForConditionalGeneration,
    BlipProcessor,
    CLIPModel,
    CLIPProcessor,
)


@dataclass
class ProductPrediction:
    """Structured Step 5 output for a single product image."""
    category: str
    category_confidence: float
    top_k_categories: List[Tuple[str, float]]
    description: str


class MultimodalProductPipeline:
    """Wraps CLIP (classification) and BLIP (captioning) behind one .predict() call."""

    def __init__(
        self,
        clip_model_name: str = "openai/clip-vit-base-patch32",
        blip_model_name: str = "Salesforce/blip-image-captioning-base",
        device: str = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # --- CLIP: shared image/text embedding space, used as the classification head ---
        self.clip_model = CLIPModel.from_pretrained(clip_model_name).to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)

        # --- BLIP: image captioning model, used as the description generator ---
        self.blip_processor = BlipProcessor.from_pretrained(blip_model_name)
        self.blip_model = BlipForConditionalGeneration.from_pretrained(blip_model_name).to(self.device)

        self.clip_model.eval()
        self.blip_model.eval()

    @torch.no_grad()
    def classify(
        self, image: Image.Image, categories: List[str], top_k: int = 3
    ) -> Tuple[str, float, List[Tuple[str, float]]]:
        """Step 4a: zero-shot category prediction via CLIP.

        Scores the image against `a photo of a {category}` prompts for every
        candidate category and returns a softmax-normalized ranking.
        """
        if not categories:
            raise ValueError("categories list must not be empty")

        prompts = [f"a photo of a {c}" for c in categories]
        inputs = self.clip_processor(
            text=prompts, images=image, return_tensors="pt", padding=True
        ).to(self.device)

        outputs = self.clip_model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1).squeeze(0).cpu()

        ranked = sorted(zip(categories, probs.tolist()), key=lambda x: x[1], reverse=True)
        top_category, top_prob = ranked[0]
        return top_category, top_prob, ranked[:top_k]

    @torch.no_grad()
    def describe(self, image: Image.Image, max_new_tokens: int = 30) -> str:
        """Step 4b: natural-language description via BLIP."""
        inputs = self.blip_processor(image, return_tensors="pt").to(self.device)
        output_ids = self.blip_model.generate(**inputs, max_new_tokens=max_new_tokens)
        return self.blip_processor.decode(output_ids[0], skip_special_tokens=True)

    def predict(self, image: Image.Image, categories: List[str]) -> ProductPrediction:
        """Steps 2-4 combined: run the full pipeline on one product image."""
        image = image.convert("RGB")  # Step 2: minimal preprocessing
        category, confidence, top_k = self.classify(image, categories)
        description = self.describe(image)
        return ProductPrediction(
            category=category,
            category_confidence=confidence,
            top_k_categories=top_k,
            description=description,
        )
