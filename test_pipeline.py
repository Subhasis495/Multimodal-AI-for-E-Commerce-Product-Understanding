"""
Lightweight tests for the pipeline's *logic* (ranking, preprocessing, catalog
I/O) using mocked CLIP/BLIP models — no internet access or model download
required. Useful as a smoke test before running the real Streamlit app.

Run with:
    python -m unittest test_pipeline.py -v
"""

import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import torch
from PIL import Image

from model_pipeline import MultimodalProductPipeline


def _fake_clip_outputs(logits):
    """Build a fake CLIP model output object with a fixed logits_per_image."""
    out = MagicMock()
    out.logits_per_image = torch.tensor([logits])
    return out


class _FakeBatch(dict):
    """Mimics HF's BatchEncoding: a dict that also supports .to(device)."""

    def to(self, device):
        return self


class TestClassify(unittest.TestCase):
    def setUp(self):
        # Skip the real __init__ (which would download models) and build the
        # object manually, then inject mock model/processor attributes.
        self.pipeline = MultimodalProductPipeline.__new__(MultimodalProductPipeline)
        self.pipeline.device = "cpu"

        fake_batch = _FakeBatch(
            input_ids=torch.tensor([[0]]), pixel_values=torch.zeros(1, 3, 2, 2)
        )
        self.pipeline.clip_processor = MagicMock(return_value=fake_batch)
        self.pipeline.clip_model = MagicMock()

    def test_classify_picks_highest_probability_category(self):
        categories = ["sneakers", "t-shirt", "laptop"]
        # Highest logit should be "t-shirt" (index 1)
        self.pipeline.clip_model.return_value = _fake_clip_outputs([1.0, 5.0, 2.0])

        image = Image.new("RGB", (4, 4))
        top_cat, top_prob, ranked = self.pipeline.classify(image, categories, top_k=2)

        self.assertEqual(top_cat, "t-shirt")
        self.assertEqual(len(ranked), 2)
        expected_probs = torch.tensor([1.0, 5.0, 2.0]).softmax(0).tolist()
        self.assertAlmostEqual(top_prob, max(expected_probs), places=4)

    def test_classify_raises_on_empty_categories(self):
        image = Image.new("RGB", (4, 4))
        with self.assertRaises(ValueError):
            self.pipeline.classify(image, [])


class TestCatalog(unittest.TestCase):
    def test_save_and_load_roundtrip(self):
        import catalog as catalog_module

        with tempfile.TemporaryDirectory() as tmpdir:
            fake_path = os.path.join(tmpdir, "product_catalog.json")
            with patch.object(catalog_module, "CATALOG_PATH", fake_path):
                entry = catalog_module.save_to_catalog(
                    image_filename="shoe.jpg",
                    category="running shoes",
                    confidence=0.873,
                    description="a pair of red running shoes",
                )
                loaded = catalog_module.load_catalog()

                self.assertEqual(len(loaded), 1)
                self.assertEqual(loaded[0]["category"], "running shoes")
                self.assertEqual(loaded[0]["image_filename"], "shoe.jpg")
                self.assertEqual(entry, loaded[0])

                # File should be valid JSON on disk too
                with open(fake_path) as f:
                    raw = json.load(f)
                self.assertEqual(raw, loaded)


class TestEvaluate(unittest.TestCase):
    def test_classify_accuracy(self):
        from evaluate import classify_accuracy

        preds = ["shoes", "shirt", "laptop"]
        truth = ["shoes", "jacket", "laptop"]
        self.assertAlmostEqual(classify_accuracy(preds, truth), 2 / 3)

    def test_top_k_accuracy(self):
        from evaluate import top_k_accuracy

        preds = [["shoes", "boots"], ["shirt", "jacket"]]
        truth = ["boots", "t-shirt"]
        self.assertAlmostEqual(top_k_accuracy(preds, truth), 0.5)

    def test_simple_bleu1(self):
        from evaluate import simple_bleu1

        self.assertAlmostEqual(simple_bleu1("a red shoe", "a red running shoe"), 1.0)
        self.assertAlmostEqual(simple_bleu1("blue hat", "a red running shoe"), 0.0)


if __name__ == "__main__":
    unittest.main()
