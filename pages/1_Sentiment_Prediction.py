import streamlit as st
from ml.sentiment_ml import MLSentiment
from ml.bert_sentiment import BERTSentiment

ml = MLSentiment()
bert = BERTSentiment()

st.title("🔍 Sentiment Prediction (ML + BERT)")

text = st.text_area(
    "Enter Research Abstract",
    key="agentic_text_input"
)


if st.button("Predict"):
    ml_sent, confidence, lr, rf = ml.predict(text)
    bert_sent = bert.predict(text)

    st.write("Logistic Regression:", lr)
    st.write("Random Forest:", rf)
    st.write("BERT:", bert_sent)
    st.success(f"Final Sentiment: {ml_sent.upper()} | Confidence: {confidence}")
