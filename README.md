# Research-Paper-Abstract-Sentiment-Analysis-By-Machine-Learning-
<!DOCTYPE html>
<html lang="en">
<head>

  
</head>

<body>

<header>
    <h1>Research Abstract Sentiment Analysis</h1>
    <p>Machine Learning & BERT-based NLP Project</p>
</header>

<section>
    <h2>📌 Project Overview</h2>
    <p>
        This project implements an automated system to analyze the sentiment of
        research paper abstracts using Natural Language Processing (NLP) and
        Machine Learning techniques. The abstracts are classified into
        <strong>Positive</strong>, <strong>Neutral</strong>, or <strong>Negative</strong>
        categories to assist researchers during literature reviews.
    </p>
</section>

<section>
    <h2>🎯 Objectives</h2>
    <ul>
        <li>Automate sentiment analysis of academic abstracts</li>
        <li>Reduce manual effort during literature survey</li>
        <li>Compare classical ML models with transformer models</li>
        <li>Provide an interactive web interface</li>
    </ul>
</section>

<section>
    <h2>🧠 Models Implemented</h2>

<h3>Logistic Regression (TF-IDF)</h3>
    <ul>
        <li>Simple and interpretable</li>
        <li>Accuracy ≈ 72%</li>
    </ul>

 <h3>Random Forest (TF-IDF)</h3>
    <ul>
        <li>Ensemble-based classifier</li>
        <li>Accuracy ≈ 73%</li>
    </ul>

<h3>BERT (Transformer Model)</h3>
    <ul>
        <li>Model: <code>bert-base-uncased</code></li>
        <li>Context-aware sentiment understanding</li>
        <li>Accuracy ≈ 80–85%</li>
    </ul>
</section>

<section>
    <h2>🔄 System Workflow</h2>
    <pre>
User Input (Abstract)
        ↓
Text Preprocessing
        ↓
TF-IDF / BERT Tokenization
        ↓
Model Prediction
        ↓
Sentiment Output
    </pre>
</section>

<section>
    <h2>🛠️ Technologies Used</h2>
    <ul>
        <li>Python</li>
        <li>NLTK, TF-IDF</li>
        <li>Scikit-learn</li>
        <li>PyTorch</li>
        <li>Hugging Face Transformers</li>
        <li>Streamlit</li>
    </ul>
</section>

<section>
    <h2>📂 Project Structure</h2>
    <pre>
Research-Abstract-Sentiment-Analysis/
│
├── app.py
├── requirements.txt
├── README.html
│
├── bert_model/
├── logistic_regression_model.pkl
├── random_forest_model.pkl
├── tfidf_vectorizer.pkl
└── arxiv_sample.csv
    </pre>
</section>

<section>
    <h2>🚀 How to Run</h2>
    <pre>
conda create -n research_sentiment python=3.10
conda activate research_sentiment
pip install -r requirements.txt
streamlit run app.py
    </pre>
</section>

<section>
    <h2>📈 Results</h2>
    <table>
        <tr>
            <th>Model</th>
            <th>Accuracy</th>
        </tr>
        <tr>
            <td>Logistic Regression</td>
            <td>~72%</td>
        </tr>
        <tr>
            <td>Random Forest</td>
            <td>~73%</td>
        </tr>
        <tr>
            <td><strong>BERT</strong></td>
            <td><strong>~80–85%</strong></td>
        </tr>
    </table>
</section>

<section>
    <h2>🔮 Future Scope</h2>
    <ul>
        <li>PDF-based full paper sentiment analysis</li>
        <li>SciBERT integration</li>
        <li>Summarization + sentiment</li>
        <li>Deployment on Hugging Face Spaces</li>
    </ul>
</section>

<section>
    <h2>👩‍💻 Author</h2>
    <p>
        <strong>Sparshi Jain</strong><br>
        B.Tech – Mathematics & Computing<br>
        Minor Project – Machine Learning & NLP
    </p>
</section>
<section>
    <h2>🏗️ System Architecture</h2>
    <p>
        The architecture of the Research Abstract Sentiment Analysis system follows
        a modular pipeline-based design. Each component is independent and contributes
        to efficient sentiment prediction.
    </p>

 <pre>
┌──────────────────────────┐
│        User Input        │
│  (Research Abstract)     │
└────────────┬─────────────┘
             ↓
┌──────────────────────────┐
│   Text Preprocessing     │
│ - Lowercasing            │
│ - Stopword Removal       │
│ - Tokenization           │
│ - Cleaning               │
└────────────┬─────────────┘
             ↓
┌─────────────────────────────────────┐
│        Feature Engineering           │
│ ┌───────────────┐  ┌──────────────┐ │
│ │ TF-IDF Vector │  │ BERT Tokenizer│ │
│ └───────────────┘  └──────────────┘ │
└────────────┬────────────────────────┘
             ↓
┌─────────────────────────────────────┐
│       Model Selection Layer         │
│ ┌──────────────┐ ┌───────────────┐ │
│ │ Logistic Reg │ │ Random Forest │ │
│ └──────────────┘ └───────────────┘ │
│            ┌────────────────────┐ │
│            │        BERT         │ │
│            └────────────────────┘ │
└────────────┬────────────────────────┘
             ↓
┌──────────────────────────┐
│   Sentiment Prediction   │
│ (Positive / Neutral /    │
│        Negative)         │
└────────────┬─────────────┘
             ↓
┌──────────────────────────┐
│  Streamlit Web Interface │
│   (Result Display)       │
└──────────────────────────┘
    </pre>
</section>

<section>
    <h3>Random Forest – Confusion Matrix</h3>
    <p>
        The confusion matrix below illustrates the classification performance
        of the Random Forest model across positive, neutral, and negative classes.
    </p>

 <img src=".png"
         alt="Random Forest Confusion Matrix"
         style="width:80%; max-width:700px; display:block; margin:auto;">

<p style="text-align:center; font-style:italic;">
        Figure 1: Confusion Matrix for Random Forest Model
    </p>
</section>
<section>
    <h3>Logistic Regression – Confusion Matrix</h3>
    <p>
        This confusion matrix represents the performance of the Logistic Regression
        classifier on the test dataset.
    </p>

 <img src="images/lr_confusion_matrix.png"
         alt="Logistic Regression Confusion Matrix"
         style="width:80%; max-width:700px; display:block; margin:auto;">

<p style="text-align:center; font-style:italic;">
        Figure 2: Confusion Matrix for Logistic Regression Model
    </p>
</section>

<section>
    <h3>Model Accuracy Comparison</h3>

  <img src="images/model_accuracy.png"
         alt="Model Accuracy Comparison"
         style="width:70%; max-width:600px; display:block; margin:auto;">

<p style="text-align:center; font-style:italic;">
        Figure 3: Accuracy comparison between Logistic Regression, Random Forest, and BERT
    </p>
</section>





<footer>
    <p>© 2025 Research Abstract Sentiment Analysis</p>
</footer>

</body>
</html>

