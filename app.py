import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# ==============================
# LOAD YOUR OWN MODEL (from GitHub folder)
# ==============================
@st.cache_resource(show_spinner="Loading models...")
def load_models():
    # YOUR LOCAL MODEL (now on GitHub)
    sentiment_analyzer = pipeline(
        "text-classification",
        model="final_starbucks_model",
        device=-1
    )

    # STRONG AI MODEL (online, works everywhere)
    model_name = "MBZUAI/LaMini-Flan-T5-248M"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    return sentiment_analyzer, tokenizer, model

sentiment_analyzer, tokenizer, model = load_models()

# ==============================
# CORE FUNCTION (Step 4 logic)
# ==============================
def analyze_customer_review(review):
    # 1. Sentiment
    sentiment = sentiment_analyzer(review)[0]["label"]

    # 2. Summary
    summary_prompt = f"summarize customer review in one short sentence: {review}"
    summary_ids = model.generate(
        **tokenizer(summary_prompt, return_tensors="pt", truncation=True, max_length=512),
        max_length=30,
        min_length=8,
        num_beams=4,
        do_sample=False,
        repetition_penalty=1.2
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    # 3. Service reply
    reply_prompt = (
        f"Customer feedback summary: {summary}\n"
        "Write a polite Starbucks customer service reply starting with: Thank you for your valuable feedback."
    )
    reply_ids = model.generate(
        **tokenizer(reply_prompt, return_tensors="pt", truncation=True, max_length=512),
        max_length=50,
        min_length=20,
        num_beams=5,
        do_sample=False,
        repetition_penalty=1.2
    )
    reply = tokenizer.decode(reply_ids[0], skip_special_tokens=True)

    return sentiment, summary, reply

# ==============================
# STREAMLIT INTERFACE
# ==============================
st.title("Starbucks Customer Review Analyzer")

review_input = st.text_area("Enter the customer review:")

if st.button("Analyze Review"):
    if review_input:
        sentiment, summary, reply = analyze_customer_review(review_input)

        st.subheader("Sentiment")
        st.write(sentiment)

        st.subheader("Summary")
        st.write(summary)

        st.subheader("Generated Customer Service Reply")
        st.write(reply)
    else:
        st.warning("Please enter a review first!")
