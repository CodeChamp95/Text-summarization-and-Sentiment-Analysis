import streamlit as st
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification, AutoModelForSeq2SeqLM
import tensorflow as tf
from tensorflow import keras
import numpy as np

# Sentiment mapping
sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}

# Streamlit UI
st.set_page_config(page_title="Text Analyzer", layout="centered")
st.title("\ud83d\udcc8 Text Summarizer & Sentiment Analyzer")
st.markdown("Type or paste any block of text below. First generate a summary, then analyze its sentiment.")

input_text = st.text_area("Enter content:", height=300)

if "summary" not in st.session_state:
    st.session_state.summary = ""

if st.button("Generate Summary"):
    if input_text.strip() == "":
        st.warning("Please enter some text before generating summary.")
    else:
        with st.spinner("Generating summary..."):
            bart_tokenizer = AutoTokenizer.from_pretrained("CodeChamp95/bart_financial_tokenizer")
            bart_model = AutoModelForSeq2SeqLM.from_pretrained("CodeChamp95/bart_summary_financial_model")
            inputs = bart_tokenizer([input_text], return_tensors="pt", max_length=512, truncation=True)
            summary_ids = bart_model.generate(inputs["input_ids"], num_beams=4, max_length=150, early_stopping=True)
            st.session_state.summary = bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

if st.session_state.summary:
    st.subheader("Summary")
    st.success(st.session_state.summary)

    if st.button("Analyze Sentiment of Summary"):
        with st.spinner("Analyzing sentiment..."):
            bert_tokenizer = AutoTokenizer.from_pretrained("CodeChamp95/bert_financial_tokenizer")
            bert_model = TFAutoModelForSequenceClassification.from_pretrained("CodeChamp95/bert_sentiment_financial_model")
            tokens = bert_tokenizer(st.session_state.summary, return_tensors="tf", padding=True, truncation=True, max_length=128)
            output = bert_model(tokens)
            sentiment_score = tf.nn.softmax(output.logits, axis=-1).numpy()[0]
            sentiment_label = sentiment_map[np.argmax(sentiment_score)]

        st.subheader("Sentiment")
        st.info(f"**{sentiment_label}** (Confidence: {np.max(sentiment_score)*100:.2f}%)")

        st.caption("Model: BART for summarization, BERT for sentiment classification")
