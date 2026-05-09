import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
from transformers import pipeline

# ==============================================
# ✅ 同时加载 YOUR 两个模型！！
# ==============================================
@st.cache_resource(show_spinner="Loading YOUR TWO models...")
def load_both_models():
    # 你的模型 1
    model1 = pipeline(
        "text-classification",
        model="Nancyaaaaaaa/starbucks_sentiment_model",
        device=-1
    )

    # 你的模型 2
    model2 = pipeline(
        "text-classification",
        model="Nancyaaaaaaa/final_starbucks_model",
        device=-1
    )
    return model1, model2

# 加载！
sentiment_model1, sentiment_model2 = load_both_models()

# ==============================================
# 分析功能（两个模型一起输出）
# ==============================================
def analyze(review):
    res1 = sentiment_model1(review)[0]
    res2 = sentiment_model2(review)[0]
    return res1, res2

# ==============================================
# 界面
# ==============================================
st.title("✅ MY TWO TRAINED SENTIMENT MODELS")
st.subheader("Starbucks Review Analyzer")

review = st.text_area("Enter review:")

if st.button("Analyze"):
    if review:
        r1, r2 = analyze(review)

        st.success("Model 1: starbucks_sentiment_model")
        st.write(f"Sentiment: {r1['label']} | Score: {r1['score']:.3f}")

        st.success("Model 2: final_starbucks_model")
        st.write(f"Sentiment: {r2['label']} | Score: {r2['score']:.3f}")

st.caption("✅ Both models are trained by ME")
