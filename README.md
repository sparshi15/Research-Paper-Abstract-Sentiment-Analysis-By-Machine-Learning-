# Research-Paper-Abstract-Sentiment-Analysis-By-Machine-Learning-
<!DOCTYPE html>
<html lang="en">
<head>


<body>

<header>
    <h1>Research Abstract Sentiment Analysis</h1>
    <p>Machine Learning & NLP-based Academic Text Analysis</p>
</header>

<section>
    <h2>📌 Project Overview</h2>
    <p>
        This project implements an automated system to analyze the sentiment of
        research paper abstracts using Machine Learning and Natural Language Processing.
        The system classifies abstracts into <strong>Positive</strong>,
        <strong>Neutral</strong>, or <strong>Negative</strong> categories to help
        researchers quickly prioritize papers during literature surveys.
    </p>
</section>

<section>
    <h2>🎯 Objectives</h2>
    <ul>
        <li>Automate sentiment classification of research abstracts</li>
        <li>Reduce manual effort in literature review</li>
        <li>Compare classical ML models with deep learning models</li>
        <li>Provide a simple web-based interface</li>
    </ul>
</section>

<section>
    <h2>🧠 Technologies Used</h2>
    <ul>
        <li><strong>Language:</strong> Python</li>
        <li><strong>NLP:</strong> NLTK, TF-IDF</li>
        <li><strong>ML Models:</strong> Logistic Regression, Random Forest</li>
        <li><strong>Deep Learning:</strong> BERT (Transformers)</li>
        <li><strong>Framework:</strong> Scikit-learn, Hugging Face</li>
        <li><strong>Web App:</strong> Streamlit</li>
        <li><strong>Dataset:</strong> arXiv abstracts (Kaggle)</li>
    </ul>
</section>

<section>
    <h2>🔄 System Workflow</h2>
    <p>
        Abstract Text → Preprocessing → Feature Extraction →
        Model Prediction → Sentiment Output
    </p>
</section>

<section>
    <h2>📊 Models Implemented</h2>

   Logistic Regression</h3>
    <ul>
        <li>Fast and interpretable</li>
        <li>Uses TF-IDF features</li>
        <li>Accuracy ≈ 72%</li>
    </ul>

  <h3>2. Random Forest</h3>
    <ul>
        <li>Ensemble of decision trees</li>
        <li>Handles non-linearity</li>
        <li>Accuracy ≈ 73%</li>
    </ul>
</section>

<section>
    <h2>🤖 BERT-Based Sentiment Analysis</h2>
    <p>
        BERT (Bidirectional Encoder Representations from Transformers) is a deep
        learning model that understands contextual meaning in text. It is particularly
        effective for scientific and academic language where sentiment is subtle.
    </p>

  <ul>
        <li><strong>Model:</strong> bert-base-uncased</li>
        <li><strong>Architecture:</strong> Transformer Encoder</li>
        <li><strong>Classification:</strong> Positive / Neutral / Negative</li>
        <li><strong>Fine-Tuned:</strong> On arXiv research abstracts</li>
    </ul>

   <h3>BERT Workflow</h3>
    <p>
        Text → BERT Tokenizer → Pre-trained BERT →
        Classification Head → Sentiment Output
    </p>

   <h3>Sample BERT Code</h3>
    <pre>
from transformers import BertTokenizer, BertForSequenceClassification
import torch

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=3
)

text = "This paper shows significant improvement in performance."
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

outputs = model(**inputs)
prediction = torch.argmax(outputs.logits, dim=1)
print(prediction)
    </pre>
</section>

<section>
    <h2>📈 Model Performance Comparison</h2>
    <table>
        <tr>
            <th>Model</th>
            <th>Accuracy</th>
            <th>Remarks</th>
        </tr>
        <tr>
            <td>Logistic Regression</td>
            <td>~72%</td>
            <td>Stable and interpretable</td>
        </tr>
        <tr>
            <td>Random Forest</td>
            <td>~73%</td>
            <td>Slight bias toward majority class</td>
        </tr>
        <tr>
            <td><strong>BERT</strong></td>
            <td><strong>~80–85%</strong></td>
            <td>Best contextual understanding</td>
        </tr>
    </table>
</section>

<section>
    <h2>🖥️ Streamlit Web Interface</h2>
    <ul>
        <li>User pastes abstract text</li>
        <li>Model processes input</li>
        <li>Sentiment prediction is displayed</li>
    </ul>
</section>

<section>
    <h2>🔮 Future Scope</h2>
    <ul>
        <li>Full research paper sentiment analysis</li>
        <li>Use SciBERT for scientific text</li>
        <li>PDF upload and processing</li>
        <li>Sentiment + summarization</li>
        <li>Integration with academic databases</li>
    </ul>
</section>

<section>
    <h2>👩‍💻 Author</h2>
    <p>
        <strong>Sparshi Jain</strong><br>
        B.Tech – Mathematics & Computing<br>
        Minor Project (Machine Learning & NLP)
    </p>
</section>

<footer>
    <p>© 2025 Research Abstract Sentiment Analysis Project</p>
</footer>

</body>
</html>
