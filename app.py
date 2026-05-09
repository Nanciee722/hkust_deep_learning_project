import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# --------------------------
# 模型加载（绝对不报错版本）
# --------------------------
@st.cache_resource(show_spinner="Loading models...")
def load_models():
    # 直接用公开模型 → 绝对能跑！
    sentiment_analyzer = pipeline(
        "text-classification",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=-1
    )

    # 摘要 + 回复模型
    gen_name = "MBZUAI/LaMini-Flan-T5-248M"
    tokenizer = AutoTokenizer.from_pretrained(gen_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(gen_name)

    return sentiment_analyzer, tokenizer, model

sentiment_analyzer, tokenizer, model = load_models()

# --------------------------
# 分析函数
# --------------------------
def analyze_review(review):
    sentiment = sentiment_analyzer(review)[0]["label"]

    # 摘要
    sum_prompt = f"summarize customer review in one sentence: {review}"
    sum_ids = model.generate(
        **tokenizer(sum_prompt, return_tensors="pt", truncation=True),
        max_length=30, min_length=8, num_beams=4
    )
    summary = tokenizer.decode(sum_ids[0], skip_special_tokens=True)

    # 回复
    reply_prompt = f"""Customer summary: {summary}
    Write a polite Starbucks service reply starting with: Thank you for your valuable feedback."""

    reply_ids = model.generate(
        **tokenizer(reply_prompt, return_tensors="pt", truncation=True),
        max_length=50, min_length=20, num_beams=5
    )
    reply = tokenizer.decode(reply_ids[0], skip_special_tokens=True)

    return sentiment, summary, reply

# --------------------------
# 界面
# --------------------------
st.title("Starbucks Review Analyzer")

review = st.text_area("Enter your review:")

if st.button("Analyze"):
    if review:
        sentiment, summary, reply = analyze_review(review)
        st.subheader("Sentiment")
        st.write(sentiment)
        st.subheader("Summary")
        st.write(summary)
        st.subheader("Service Reply")
        st.write(reply)
    else:
        st.warning("Please enter a review.")
