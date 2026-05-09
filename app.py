import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# --------------------------
# 直接加载本地模型！！！
# --------------------------
sentiment_analyzer = pipeline(
    "text-classification",
    model="./final_starbucks_model",  # 直接读你本地的文件夹
    device=-1
)

# 摘要和回复模型
gen_model_name = "MBZUAI/LaMini-Flan-T5-248M"
gen_tokenizer = AutoTokenizer.from_pretrained(gen_model_name)
gen_model = AutoModelForSeq2SeqLM.from_pretrained(gen_model_name)

# --------------------------
# 测试函数
# --------------------------
def analyze_review(review):
    sentiment = sentiment_analyzer(review)[0]["label"]

    summary_prompt = f"summarize customer review in one short sentence: {review}"
    summary_ids = gen_model.generate(
        **gen_tokenizer(summary_prompt, return_tensors="pt", truncation=True, max_length=512),
        max_length=30, min_length=8, num_beams=4, do_sample=False
    )
    summary = gen_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    reply_prompt = (
        f"Customer feedback summary: {summary}\n"
        "Write a polite Starbucks customer service reply starting with: Thank you for your valuable feedback."
    )
    reply_ids = gen_model.generate(
        **gen_tokenizer(reply_prompt, return_tensors="pt", truncation=True, max_length=512),
        max_length=50, min_length=20, num_beams=5, do_sample=False
    )
    reply = gen_tokenizer.decode(reply_ids[0], skip_special_tokens=True)

    return sentiment, summary, reply

# --------------------------
# 本地测试
# --------------------------
if __name__ == "__main__":
    print("=== Starbucks Review Analyzer (Local Version) ===")
    test_review = "I really enjoy the rich coffee flavor and cozy atmosphere at this Starbucks. I am quite disappointed with the extremely long waiting time during busy hours."
    
    sentiment, summary, reply = analyze_review(test_review)
    
    print("\n--- Customer Review ---")
    print(test_review)
    print("\n--- Sentiment ---")
    print(sentiment)
    print("\n--- Summary ---")
    print(summary)
    print("\n--- Generated Reply ---")
    print(reply)
