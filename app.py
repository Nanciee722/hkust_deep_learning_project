import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# --------------------------
# Load your private models from Hugging Face
# --------------------------
@st.cache_resource(show_spinner="Loading your model...")
def load_models():
    # Choose which sentiment model to use (update this line to switch models)
    sentiment_model_name = "Nancyaaaaaaa/starbucks_sentiment_model"
    # sentiment_model_name = "Nancyaaaaaaa/final_starbucks_model"  # Uncomment this line to use the other model

    sentiment_analyzer = pipeline(
        "text-classification",
        model=sentiment_model_name,
        use_auth_token=st.secrets["HF_TOKEN"],
        device=-1
    )

    # Model for summary and reply generation
    gen_model_name = "MBZUAI/LaMini-Flan-T5-248M"
    tokenizer = AutoTokenizer.from_pretrained(gen_model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(gen_model_name)

    return sentiment_analyzer, tokenizer, model

sentiment_analyzer, tokenizer, model = load_models()

# --------------------------
# Core analysis function
# --------------------------
def analyze_customer_review(review):
    # 1. Sentiment analysis
    sentiment = sentiment_analyzer(review)[0]["label"]

    # 2. Generate summary
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

    # 3. Generate customer service reply
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

# --------------------------
# Streamlit UI
# --------------------------
st.title("Starbucks Customer Review Analyzer ☕")

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
