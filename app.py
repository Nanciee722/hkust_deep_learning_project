import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

@st.cache_resource(show_spinner="Loading models...")
def load_models():
    # 公开在线情感模型（不用上传任何文件）
    sentiment_analyzer = pipeline(
        "text-classification",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=-1
    )

    # 在线摘要+回复模型
    model_name = "MBZUAI/LaMini-Flan-T5-248M"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    return sentiment_analyzer, tokenizer, model

sentiment_analyzer, tokenizer, model = load_models()

def analyze_customer_review(review):
    sentiment = sentiment_analyzer(review)[0]["label"]

    summary_prompt = f"summarize customer review in one short sentence: {review}"
    summary_ids = model.generate(
        **tokenizer(summary_prompt, return_tensors="pt", truncation=True, max_length=512),
        max_length=30, min_length=8, num_beams=4, repetition_penalty=1.2
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    reply_prompt = (
        f"Customer feedback summary: {summary}\n"
        "Write a polite Starbucks customer service reply starting with: Thank you for your valuable feedback."
    )
    reply_ids = model.generate(
        **tokenizer(reply_prompt, return_tensors="pt", truncation=True, max_length=512),
        max_length=50, min_length=20, num_beams=5, repetition_penalty=1.2
    )
    reply = tokenizer.decode(reply_ids[0], skip_special_tokens=True)

    return sentiment, summary, reply

st.title("Starbucks Review Analyzer")
review = st.text_area("Enter your review:")

if st.button("Analyze"):
    if review:
        sentiment, summary, reply = analyze_customer_review(review)
        st.subheader("Sentiment")
        st.write(sentiment)
        st.subheader("Summary")
        st.write(summary)
        st.subheader("Service Reply")
        st.write(reply)
    else:
        st.warning("Please enter a review!")
