import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
from transformers import pipeline

# ================================
# 你的两个模型（现在已经正确上传）
# ================================
@st.cache_resource(show_spinner="Loading models...")
def load_models():
    m1 = pipeline("text-classification", model="Nancyaaaaaaa/starbucks_sentiment_model")
    m2 = pipeline("text-classification", model="Nancyaaaaaaa/final_starbucks_model")
    return m1, m2

model1, model2 = load_models()

# ================================
# 界面
# ================================
st.title("✅ My Two Trained Models")
review = st.text_area("Review:")

if st.button("Analyze"):
    if review:
        r1 = model1(review)[0]
        r2 = model2(review)[0]

        st.subheader("Model 1")
        st.write(r1["label"], round(r1["score"], 3))

        st.subheader("Model 2")
        st.write(r2["label"], round(r2["score"], 3))

st.caption("All models trained by ME")
