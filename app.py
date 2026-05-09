import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# --------------------------
# 只加载 1 个你的模型（稳定！）
# --------------------------
@st.cache_resource(show_spinner="Loading your trained model...")
def load_my_model():
    return pipeline(
        "text-classification",
        model="Nancyaaaaaaa/final_starbucks_model"
    )

# 加载你的模型
sentiment_model = load_my_model()

# 摘要 + 回复模型
gen_tokenizer = AutoTokenizer.from_pretrained("MBZUAI/LaMini-Flan-T5-248M")
gen_model = AutoModelForSeq2SeqLM.from_pretrained("MBZUAI/LaMini-Flan-T5-248M")

# --------------------------
# 功能
# --------------------------
def analyze(review):
    sent = sentiment_model(review)[0]["label"]
    
    # 摘要
    sum_ids = gen_model.generate(
        **gen_tokenizer(f"summarize: {review}", return_tensors="pt"),
        max_length=30
    )
    summary = gen_tokenizer.decode(sum_ids[0], skip_special_tokens=True)
    
    # 回复
    reply_ids = gen_model.generate(
        **gen_tokenizer(f"reply politely: {review}", return_tensors="pt"),
        max_length=60
    )
    reply = gen_tokenizer.decode(reply_ids[0], skip_special_tokens=True)
    
    return sent, summary, reply

# --------------------------
# 界面
# --------------------------
st.title("Starbucks Review Analyzer")
st.subheader("✅ Using MY OWN TRAINED MODEL")

review = st.text_area("Enter your review:")

if st.button("Analyze"):
    if review:
        sentiment, summary, reply = analyze(review)
        st.success(f"Sentiment: {sentiment}")
        st.info(f"Summary: {summary}")
        st.chat_message("assistant").write(reply)

st.caption("Model trained by ME | Homework Completed")
