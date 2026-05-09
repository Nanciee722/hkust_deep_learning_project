import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
from transformers import pipeline


# ── Load model ─────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading sentiment model...")
def load_model():
    """
    Load the fine-tuned Starbucks sentiment model from Hugging Face Hub.
    The model files are stored inside a subfolder with the same name as the repo.
    """
    model = pipeline(
        "text-classification",
        model="Nancyaaaaaaa/starbucks_sentiment_model",
        subfolder="starbucks_sentiment_model",
        device=-1  # force CPU
    )
    return model

sentiment_model = load_model()


# ── Analysis function ──────────────────────────────────────────────────────────
def analyze(review: str) -> dict:
    """
    Run sentiment classification on the given review text.

    Args:
        review: Customer review string.

    Returns:
        Dict with 'label' and 'score' keys.
    """
    result = sentiment_model(review)[0]
    return result


# ── UI ─────────────────────────────────────────────────────────────────────────
st.title("☕ Starbucks Review Sentiment Analyzer")
st.subheader("Powered by my fine-tuned model")

review = st.text_area("Enter a Starbucks review:")

if st.button("Analyze"):
    if review.strip():
        result = analyze(review)
        st.success(f"Sentiment: **{result['label']}**")
        st.write(f"Confidence: {result['score']:.3f}")
    else:
        st.warning("Please enter a review first.")

st.caption("✅ Model trained by Nancyaaaaaaa")
