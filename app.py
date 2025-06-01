import streamlit as st
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification, AutoModelForSeq2SeqLM
import tensorflow as tf
from tensorflow import keras
import numpy as np

# Load BART for summarization
bart_model = AutoModelForSeq2SeqLM.from_pretrained("bart_summary_financial_model")
bart_tokenizer = AutoTokenizer.from_pretrained("bart_financial_tokenizer")

# Load BERT for sentiment analysis
bert_model = TFAutoModelForSequenceClassification.from_pretrained("bert_sentiment_financial_model")
bert_tokenizer = AutoTokenizer.from_pretrained("bert_financial_tokenizer")

# Sentiment mapping
sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}

# Streamlit UI
st.set_page_config(page_title="Text Analyzer", layout="centered")
st.title("📈 Text Summarizer & Sentiment Analyzer")
st.markdown("Type or paste any block of text below. The app will summarize it and analyze its sentiment.")

input_text = st.text_area("Enter content:", height=300)

if st.button("Analyze"):
    if input_text.strip() == "":
        st.warning("Please enter some text before clicking Analyze.")
    else:
        with st.spinner("Generating summary and analyzing sentiment..."):
            # Summarization
            inputs = bart_tokenizer([input_text], return_tensors="pt", max_length=512, truncation=True)
            summary_ids = bart_model.generate(inputs["input_ids"], num_beams=4, max_length=150, early_stopping=True)
            summary = bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

            # Sentiment analysis
            tokens = bert_tokenizer(summary, return_tensors="tf", padding=True, truncation=True, max_length=128)
            output = bert_model(tokens)
            sentiment_score = tf.nn.softmax(output.logits, axis=-1).numpy()[0]
            sentiment_label = sentiment_map[np.argmax(sentiment_score)]

        st.subheader("Summary")
        st.success(summary)

        st.subheader("Sentiment")
        st.info(f"**{sentiment_label}** (Confidence: {np.max(sentiment_score)*100:.2f}%)")

        st.caption("Model: BART for summarization, BERT for sentiment classification")
