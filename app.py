# 本地 PyCharm 版本（用你的模型！）
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# --------------------------
# 你的本地模型（作业真正要用的！）
# --------------------------
@st.cache_resource(show_spinner="Loading YOUR model...")
def load_models():
    sentiment_analyzer = pipeline(
        "text-classification",
        model="final_starbucks_model",  # 你训练的！
        device=-1
    )

    gen_model = "MBZUAI/LaMini-Flan-T5-248M"
    tokenizer = AutoTokenizer.from_pretrained(gen_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(gen_model)

    return sentiment_analyzer, tokenizer, model

sentiment_analyzer, tokenizer, model = load_models()

def analyze_customer_review(review):
    sentiment = sentiment_analyzer(review)[0]["label"]

    summary_prompt = f"summarize customer review in one short sentence: {review}"
    summary_ids = model.generate(**tokenizer(summary_prompt, return_tensors="pt", truncation=True), max_length=30)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    reply_prompt = f"""Customer summary: {summary}
    Write a polite Starbucks service reply starting with: Thank you for your valuable feedback."""

    reply_ids = model.generate(**tokenizer(reply_prompt, return_tensors="pt", truncation=True), max_length=60)
    reply = tokenizer.decode(reply_ids[0], skip_special_tokens=True)

    return sentiment, summary, reply

st.title("Starbucks Review Analyzer (MY TRAINED MODEL)")
review = st.text_area("Enter your review:")

if st.button("Analyze"):
    if review:
        s, summa, rep = analyze_customer_review(review)
        st.subheader("Sentiment")
        st.write(s)
        st.subheader("Summary")
        st.write(summa)
        st.subheader("Service Reply")
        st.write(rep)
    else:
        st.warning("Please enter a review.")
