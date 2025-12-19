import React, { useState } from "react";
import TextForm from "./components/TextForm";
import PDFUploader from "./components/PDFUploader";
import SentimentResult from "./components/SentimentResult";
import SentimentChart from "./components/SentimentChart";
import WordCloudChart from "./components/WordCloudChart";

function App() {
  const [result, setResult] = useState(null);

  return (
    <div className="App" style={{ padding: "20px", fontFamily: "Arial" }}>
      <h1>Research Abstract Sentiment Dashboard</h1>
      
      <TextForm setResult={setResult} />
      <PDFUploader setResult={setResult} />
      
      {result && (
        <div style={{ marginTop: "20px" }}>
          <SentimentResult result={result} />
          <SentimentChart result={result} />
          <WordCloudChart words={result.top_contributing_words} />
        </div>
      )}
    </div>
  );
}

export default App;
