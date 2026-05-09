import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# --------------------------
# 加载 【你自己训练的模型】
# --------------------------
@st.cache_resource(show_spinner="Loading your trained model...")
def load_models():
    # 你的公开模型（现在不需要token了！）
    sentiment_analyzer = pipeline(
        "text-classification",
        model="Nancyaaaaaaa/starbucks_sentiment_model",
        device=-1
    )

    # 摘要 & 回复生成模型
    gen_model_name = "MBZUAI/LaMini-Flan-T5-248M"
    tokenizer = AutoTokenizer.from_pretrained(gen_model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(gen_model_name)

    return sentiment_analyzer, tokenizer, model

sentiment_analyzer, tokenizer, model = load_models()

# --------------------------
# 核心分析功能
# --------------------------
def analyze_customer_review(review):
    # 情感分析
    sentiment = sentiment_analyzer(review)[0]["label"]

    # 生成摘要
    summary_prompt = f"summarize customer review in one short sentence: {review}"
    summary_ids = model.generate(
        **tokenizer(summary_prompt, return_tensors="pt", truncation=True),
        max_length=30,
        min_length=8,
        num_beams=4
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    # 生成客服回复
    reply_prompt = f"""Customer summary: {summary}
    Write a polite Starbucks service reply starting with: Thank you for your valuable feedback."""

    reply_ids = model.generate(
        **tokenizer(reply_prompt, return_tensors="pt", truncation=True),
        max_length=50,
        min_length=20,
        num_beams=5
    )
    reply = tokenizer.decode(reply_ids[0], skip_special_tokens=True)

    return sentiment, summary, reply

# --------------------------
# 界面
# --------------------------
st.title("Starbucks Review Analyzer (My Trained Model)")

review_input = st.text_area("Enter your customer review:")

if st.button("Analyze Review"):
    if review_input.strip():
        with st.spinner("Analyzing..."):
            sentiment, summary, reply = analyze_customer_review(review_input)

        st.subheader("Sentiment")
        st.write(sentiment)

        st.subheader("Summary")
        st.write(summary)

        st.subheader("Service Reply")
        st.write(reply)
    else:
        st.warning("Please enter a review first!")
