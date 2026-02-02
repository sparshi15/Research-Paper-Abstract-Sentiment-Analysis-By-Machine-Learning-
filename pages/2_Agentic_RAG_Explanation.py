import streamlit as st
import pandas as pd
from agent.rag import RAG
from agent.agent import agent_decision

@st.cache_resource
def load_rag():
    df = pd.read_csv("data/arxiv_sample.csv")
    return RAG(df["abstract"].dropna().tolist()[:30])

rag = load_rag()

st.title("🧠 Agentic AI + RAG Explanation")

text = st.text_area(
    "Enter Research Abstract",
    key="agentic_text_input"
)

sentiment = st.selectbox("Sentiment", ["positive","neutral","negative"])
confidence = st.selectbox("Confidence", ["High","Medium","Low"])

if st.button("Explain"):
    result = agent_decision(text, sentiment, confidence, rag)

    st.subheader("Explanation")
    st.write(result["explanation"])

    with st.expander("Agent Reasoning"):
        for step in result["agent_log"]:
            st.write("•", step)

    if result["rag_used"]:
        st.success("RAG was used")


