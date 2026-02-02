import pickle
import os

class MLSentiment:
    def __init__(self):
        base = "models"
        self.tfidf = pickle.load(open(os.path.join(base, "tfidf_vectorizer.pkl"), "rb"))
        self.lr = pickle.load(open(os.path.join(base, "logistic_regression_model.pkl"), "rb"))
        self.rf = pickle.load(open(os.path.join(base, "random_forest_model.pkl"), "rb"))

    def predict(self, text):
        X = self.tfidf.transform([text])
        lr_pred = self.lr.predict(X)[0]
        rf_pred = self.rf.predict(X)[0]

        if lr_pred == rf_pred:
            return lr_pred, "High", lr_pred, rf_pred
        else:
            return lr_pred, "Medium", lr_pred, rf_pred

