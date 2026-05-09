import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# 在这里切换模型，二选一即可
# MODEL_NAME = "Nancyaaaaaaa/starbucks_sentiment_model"
MODEL_NAME = "Nancyaaaaaaa/final_starbucks_model"

@st.cache_resource(show_spinner="Loading your trained model...")
def load_models():
    sentiment_analyzer = pipeline(
        "text-classification",
        model=MODEL_NAME,
        device=-1
    )

    gen_model_name = "MBZUAI/LaMini-Flan-T5-248M"
    gen_tokenizer = AutoTokenizer.from_pretrained(gen_model_name)
    gen_model = AutoModelForSeq2SeqLM.from_pretrained(gen_model_name)

    return sentiment_analyzer, gen_tokenizer, gen_model

sentiment_analyzer, tokenizer, model = load_models()

def analyze_customer_review(review):
    sentiment = sentiment_analyzer(review)[0]["label"]

    summary_prompt = f"summarize customer review in one short sentence: {review}"
    summary_ids = model.generate(
        **tokenizer(summary_prompt, return_tensors="pt", truncation=True, max_length=512),
        max_length=30, min_length=8, num_beams=4, do_sample=False
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    reply_prompt = (
        f"Customer feedback summary: {summary}\n"
        "Write a polite Starbucks customer service reply starting with: Thank you for your valuable feedback."
    )
    reply_ids = model.generate(
        **tokenizer(reply_prompt, return_tensors="pt", truncation=True, max_length=512),
        max_length=50, min_length=20, num_beams=5, do_sample=False
    )
    reply = tokenizer.decode(reply_ids[0], skip_special_tokens=True)

    return sentiment, summary, reply

st.title("Starbucks Customer Review Analyzer ☕")
st.caption(f"Using model: {MODEL_NAME}")

review_input = st.text_area("Enter your customer review:")

if st.button("Analyze Review"):
    if review_input.strip():
        with st.spinner("Analyzing..."):
            sentiment, summary, reply = analyze_customer_review(review_input)
        st.subheader("Sentiment")
        st.write(sentiment)
        st.subheader("Summary")
        st.write(summary)
        st.subheader("Generated Service Reply")
        st.write(reply)
    else:
        st.warning("Please enter a review first.")
