import React from "react";

function SentimentResult({ result }) {
  return (
    <div style={{ marginTop: "20px" }}>
      <h3>Prediction Result</h3>
      <p><b>Abstract:</b> {result.abstract_text}</p>
      <p><b>Logistic Regression:</b> {result.predicted_sentiment}</p>
      <p><b>Random Forest:</b> {result.rf_prediction}</p>
      <p><b>Top Words:</b> {result.top_contributing_words.join(", ")}</p>
    </div>
  );
}

export default SentimentResult;
