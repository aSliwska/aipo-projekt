"""
Text classification module for road signs and billboards.
"""

import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


class TextClassifier:
    def __init__(self):
        logging.debug("Initializing TextClassifier")
        self.vectorizer = TfidfVectorizer()
        self.classifier = LogisticRegression()
        self.trained = False

    def train(self, texts, labels):
        logging.info("Training TextClassifier with texts: %s", texts)
        X_train = self.vectorizer.fit_transform(texts)
        self.classifier.fit(X_train, labels)
        self.trained = True
        logging.info("TextClassifier training complete")

    def predict(self, text):
        logging.debug("Predicting label for text: %s", text)
        if not self.trained:
            raise ValueError("Model nie został wytrenowany.")
        vect = self.vectorizer.transform([text])
        prediction = self.classifier.predict(vect)[0]
        logging.debug("Prediction result: %s", prediction)
        return prediction
