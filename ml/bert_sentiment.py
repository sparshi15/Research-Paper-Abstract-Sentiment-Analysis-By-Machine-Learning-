import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class BERTSentiment:
    def __init__(self):
        self.device = torch.device("cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(
            "nlptown/bert-base-multilingual-uncased-sentiment"
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "nlptown/bert-base-multilingual-uncased-sentiment",
            torch_dtype=torch.float32
        )
        self.model.to(self.device)
        self.model.eval()

    def predict(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.model(**inputs).logits

        idx = torch.argmax(logits, dim=1).item()
        if idx <= 1:
            return "negative"
        elif idx == 2:
            return "neutral"
        else:
            return "positive"



