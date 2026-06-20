"""
Streamlit demo app implementing the Day 1 Homework system design end-to-end:

    Product image -> preprocessing -> CLIP (category) + BLIP (description)
                   -> structured output -> saved to local product catalog

Run with:
    streamlit run app.py
"""

import streamlit as st
from PIL import Image

from catalog import load_catalog, save_to_catalog
from categories import DEFAULT_CATEGORIES
from model_pipeline import MultimodalProductPipeline

st.set_page_config(page_title="Multimodal Product Understanding", page_icon="🛍️", layout="centered")


@st.cache_resource(show_spinner="Loading CLIP + BLIP models (first run only, ~600MB download)...")
def get_pipeline() -> MultimodalProductPipeline:
    return MultimodalProductPipeline()


def main():
    st.title("🛍️ Multimodal Product Understanding")
    st.caption("Prototype — CLIP for category, BLIP for description")

    with st.sidebar:
        st.header("Settings")
        categories_text = st.text_area(
            "Category list (one per line)",
            value="\n".join(DEFAULT_CATEGORIES),
            height=240,
        )
        categories = [c.strip() for c in categories_text.split("\n") if c.strip()]
        top_k = st.slider("Top-K categories to show", min_value=1, max_value=10, value=3)

    uploaded_file = st.file_uploader("Upload a product image", type=["jpg", "jpeg", "png", "webp"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded product image", use_container_width=True)

        if st.button("Run multimodal pipeline", type="primary"):
            pipeline = get_pipeline()
            with st.spinner("Running CLIP classification + BLIP captioning..."):
                result = pipeline.predict(image, categories)

            st.subheader("Output")
            col1, col2 = st.columns(2)
            col1.metric(
                "Predicted category",
                result.category,
                f"{result.category_confidence:.1%} confidence",
            )
            col2.write("**Top candidates:**")
            for cat, prob in result.top_k_categories[:top_k]:
                col2.write(f"- {cat}: {prob:.1%}")

            st.write("**Generated description:**")
            st.success(result.description)

            save_to_catalog(
                image_filename=uploaded_file.name,
                category=result.category,
                confidence=result.category_confidence,
                description=result.description,
            )
            st.toast("Saved to product catalog", icon="✅")

    st.divider()
    st.subheader("📦 Product catalog (Step 5 output)")
    catalog = load_catalog()
    if catalog:
        st.dataframe(catalog, use_container_width=True)
    else:
        st.info("No products saved yet. Upload an image and run the pipeline above.")


if __name__ == "__main__":
    main()
