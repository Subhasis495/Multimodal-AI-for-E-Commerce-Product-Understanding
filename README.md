# Multimodal Product Understanding System

A working implementation of the system designed: given
a single product image, predict its category and generate a description,
using CLIP and BLIP together.

## How it maps to the homework design

| Homework step | Code |
|---|---|
| Step 1 — Input image | `app.py`: `st.file_uploader(...)` |
| Step 2 — Preprocessing | `model_pipeline.py`: `image.convert("RGB")` + the CLIP/BLIP processors (resize, normalize) |
| Step 3 — Multimodal model | `model_pipeline.py`: `CLIPModel` + `BlipForConditionalGeneration` |
| Step 4 — Classification head + caption generator | `MultimodalProductPipeline.classify()` (CLIP, zero-shot) and `.describe()` (BLIP) |
| Step 5 — Structured output / catalog | `catalog.py`: `save_to_catalog()` writes a JSON "database" |
| Evaluation | `evaluate.py`: `classify_accuracy`, `top_k_accuracy`, `simple_bleu1` |

## Project structure

```
product_understanding_system/
├── app.py              # Streamlit UI tying everything together
├── model_pipeline.py   # CLIP (category) + BLIP (description) pipeline
├── categories.py        # Default product category taxonomy
├── catalog.py            # Saves predictions to a local JSON "catalog"
├── evaluate.py           # Accuracy / BLEU-style evaluation helpers
├── test_pipeline.py     # Mocked unit tests (no model download needed)
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

On first run, `transformers` will download the CLIP and BLIP model weights
(~600MB combined) from Hugging Face — this requires internet access and only
happens once; they're cached locally afterward.

## Running tests (no model download needed)

`test_pipeline.py` mocks the CLIP/BLIP models so you can sanity-check the
ranking logic, catalog read/write, and evaluation metrics without downloading
anything:

```bash
python -m unittest test_pipeline.py -v
```

## Notes / things to customize

- **Categories**: edit `categories.py` (or use the sidebar in the app) to
  match your actual product taxonomy — CLIP needs no retraining for new
  categories since it's zero-shot.
- **Models**: `model_pipeline.py` defaults to `clip-vit-base-patch32` and
  `blip-image-captioning-base` for speed. Swap in larger variants
  (`clip-vit-large-patch14`, `blip-image-captioning-large`, or `BLIP-2`) for
  better accuracy at the cost of more compute.
- **Catalog storage**: `catalog.py` uses a local JSON file as a stand-in for
  a real database. For production, swap `save_to_catalog` / `load_catalog`
  for calls to Postgres, DynamoDB, etc.
- **Evaluation**: `simple_bleu1` is a minimal sanity-check metric. For a
  real evaluation report, use `sacrebleu` or `nltk` against a labeled
  validation set, as outlined in the homework's Evaluation section.
