# ==============================================================
# STREAMLIT DASHBOARD + EXPORT SENTIMENT ANALYSIS (ML + BERT)
# Author: Sparshi Jain
# ==============================================================

import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from docx import Document
from fpdf import FPDF
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# ------------------------------------------------------------
# Load Models
# ------------------------------------------------------------
@st.cache_resource
def load_ml_models():
    tfidf = pickle.load(open(r"C:\Users\91706\OneDrive\Attachments\Desktop\sen\tfidf_vectorizer.pkl", "rb"))
    lr_model = pickle.load(open(r"C:\Users\91706\OneDrive\Attachments\Desktop\sen\logistic_regression_model.pkl", "rb"))
    rf_model = pickle.load(open(r"C:\Users\91706\OneDrive\Attachments\Desktop\sen\random_forest_model.pkl", "rb"))
    return tfidf, lr_model, rf_model


@st.cache_resource
def load_bert():
    tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
    model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
    return tokenizer, model


tfidf, lr_model, rf_model = load_ml_models()
bert_tokenizer, bert_model = load_bert()


# ------------------------------------------------------------
# Prediction Functions
# ------------------------------------------------------------
def predict_tf_idf(text, model_name):
    X = tfidf.transform([text])
    return lr_model.predict(X)[0] if model_name == "Logistic Regression (TF-IDF)" else rf_model.predict(X)[0]


def predict_bert(text):
    tokens = bert_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    output = bert_model(**tokens)
    pred = torch.argmax(output.logits).item()
    return ["negative", "negative", "neutral", "positive", "positive"][pred]


# ------------------------------------------------------------
# Export Functions
# ------------------------------------------------------------
def export_to_word(df):
    doc = Document()
    doc.add_heading("Sentiment Analysis Report", level=1)

    table = doc.add_table(rows=1, cols=len(df.columns))
    table.style = "Table Grid"
    hdr = table.rows[0].cells

    for i, col in enumerate(df.columns):
        hdr[i].text = col

    for _, row in df.iterrows():
        row_cells = table.add_row().cells
        for i, value in enumerate(row):
            row_cells[i].text = str(value)

    doc.save("Sentiment_Report.docx")
    return "Sentiment_Report.docx"


def export_to_pdf(df):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=9)

    pdf.cell(200, 10, txt="Sentiment Analysis Report", ln=True, align="C")

    for i in range(len(df)):
        pdf.cell(200, 5, txt=str(df.iloc[i].to_dict()), ln=True)

    pdf.output("Sentiment_Report.pdf")
    return "Sentiment_Report.pdf"


# =============================================================
# STREAMLIT UI
# =============================================================
st.set_page_config(page_title="Research Sentiment Dashboard", layout="wide")
st.title("📊 Research Abstract Sentiment Dashboard")

st.sidebar.title("⚙ Settings")
model_choice = st.sidebar.selectbox(
    "Choose Sentiment Model",
    ["Logistic Regression (TF-IDF)", "Random Forest (TF-IDF)", "BERT (Pretrained)"]
)

mode = st.sidebar.radio("Mode", ["Upload CSV", "Manual Text Prediction"])
st.sidebar.markdown("---")

# =============================================================
# ✅ CSV Upload Prediction
# =============================================================
if mode == "Upload CSV":
    uploaded_file = st.file_uploader("📂 Upload CSV (must include 'abstract' column)", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        if "abstract" not in df.columns:
            st.error("❌ CSV must include an `abstract` column")
        else:
            if st.button("🚀 Predict Sentiment"):
                with st.spinner("Processing..."):
                    if model_choice == "BERT (Pretrained)":
                        df["Predicted Sentiment"] = df["abstract"].apply(predict_bert)
                        accuracy = None
                    else:
                        df["Predicted Sentiment"] = df["abstract"].apply(lambda x: predict_tf_idf(x, model_choice))
                        accuracy = None  # You can load saved test accuracy here if needed

                st.subheader("✅ Prediction Results")
                st.dataframe(df.head())

                # ===== Metrics =====
                st.subheader("📌 Metrics Summary")
                sentiment_counts = df["Predicted Sentiment"].value_counts(normalize=True) * 100

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Positive %", f"{sentiment_counts.get('positive', 0):.2f}%")
                col2.metric("Neutral %", f"{sentiment_counts.get('neutral', 0):.2f}%")
                col3.metric("Negative %", f"{sentiment_counts.get('negative', 0):.2f}%")
                if accuracy:
                    col4.metric("Model Accuracy", f"{accuracy:.2%}")

                # ===== Pie Chart =====
                st.subheader("🟢 Sentiment Pie Chart")
                fig, ax = plt.subplots()
                df["Predicted Sentiment"].value_counts().plot(kind="pie", autopct="%1.1f%%", ax=ax)
                ax.set_ylabel("")
                st.pyplot(fig)

                # ===== Bar Chart =====
                st.subheader("📉 Sentiment Bar Chart")
                fig2, ax2 = plt.subplots()
                df["Predicted Sentiment"].value_counts().plot(kind="bar", ax=ax2)
                st.pyplot(fig2)

                # ===== Export =====
                st.subheader("⬇ Export Results")
                st.download_button("Download CSV", df.to_csv(index=False), "sentiment_results.csv")

                if st.button("Download Word Report"):
                    file = export_to_word(df)
                    st.download_button("Download Word", open(file, "rb"), file_name=file)

                if st.button("Download PDF Report"):
                    file = export_to_pdf(df)
                    st.download_button("Download PDF", open(file, "rb"), file_name=file)


# =============================================================
# ✅ Manual Text Prediction
# =============================================================
else:
    text = st.text_area("✍ Enter Research Abstract:")

    if st.button("Predict"):
        result = predict_bert(text) if model_choice == "BERT (Pretrained)" else predict_tf_idf(text, model_choice)
        st.success(f"✅ Predicted Sentiment: **{result.upper()}**")
