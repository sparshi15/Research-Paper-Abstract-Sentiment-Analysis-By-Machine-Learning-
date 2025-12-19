import React, { useState } from "react";
import axios from "axios";

function TextForm({ setResult }) {
  const [abstract, setAbstract] = useState("");

  const handleSubmit = async () => {
    if (!abstract) return;
    try {
      const res = await axios.post("http://127.0.0.1:5000/predict", { abstract });
      setResult(res.data);
    } catch (err) {
      console.error(err);
    }
  };

  return (
    <div>
      <h2>Analyze Text Abstract</h2>
      <textarea
        rows="5"
        cols="80"
        placeholder="Paste your research abstract here..."
        value={abstract}
        onChange={(e) => setAbstract(e.target.value)}
      />
      <br />
      <button onClick={handleSubmit} style={{ marginTop: "10px" }}>Analyze</button>
    </div>
  );
}

export default TextForm;
