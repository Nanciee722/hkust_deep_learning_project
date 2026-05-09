import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# --------------------------
# ✅ 你的线上模型（Hugging Face）
# --------------------------
@st.cache_resource(show_spinner="Loading your trained model...")
def load_sentiment_model():
    return pipeline(
        "text-classification",
        model="Nancyaaaaaaa/final_starbucks_model"
    )

sentiment_model = load_sentiment_model()

# 摘要 + 回复模型
gen_tokenizer = AutoTokenizer.from_pretrained("MBZUAI/LaMini-Flan-T5-248M")
gen_model = AutoModelForSeq2SeqLM.from_pretrained("MBZUAI/LaMini-Flan-T5-248M")

# --------------------------
# 分析功能
# --------------------------
def analyze_review(review):
    sentiment = sentiment_model(review)[0]["label"]

    sum_ids = gen_model.generate(
        **gen_tokenizer(f"summarize: {review}", return_tensors="pt"),
        max_length=30
    )
    summary = gen_tokenizer.decode(sum_ids[0], skip_special_tokens=True)

    reply_ids = gen_model.generate(
        **gen_tokenizer(f"write a polite reply: {review}", return_tensors="pt"),
        max_length=60
    )
    reply = gen_tokenizer.decode(reply_ids[0], skip_special_tokens=True)

    return sentiment, summary, reply

# --------------------------
# 界面
# --------------------------
st.title("Starbucks Review Analyzer")
st.subheader("✅ Using MY TRAINED MODEL")

review = st.text_area("Enter your review:")

if st.button("Analyze"):
    if review:
        sentiment, summary, reply = analyze_review(review)
        st.success(f"Sentiment: {sentiment}")
        st.info(f"Summary: {summary}")
        st.chat_message("assistant").write(reply)

st.caption("Model trained by ME")
