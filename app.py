# app.py (Step 4 - Full AI Customer Review Analysis)
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from flask import Flask, request, jsonify, render_template
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

app = Flask(__name__)

# Load models
sentiment_analyzer = pipeline("text-classification", model="final_starbucks_model", device=-1)

model_name = "MBZUAI/LaMini-Flan-T5-248M"
tokenizer = AutoTokenizer.frompretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# AI Function (Step 4 core logic)
def analyze_customer_review(review):
    # Sentiment analysis
    sentiment = sentiment_analyzer(review)[0]["label"]

    # Summary
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

    # Customer service reply
    reply_prompt = f"""Customer feedback summary: {summary}
    Write a polite Starbucks customer service reply starting with: Thank you for your valuable feedback."""

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

# Web route
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        review_text = request.form['review']
        sentiment, summary, reply = analyze_customer_review(review_text)
        return render_template('index.html',
                               review=review_text,
                               sentiment=sentiment,
                               summary=summary,
                               reply=reply)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
