# src/models.py
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from utils import preprocess

class NaiveBayesModel:
    """
    Wrapper cho MultinomialNB kèm vectorizer.
    vectorizer_type: 'count' or 'tfidf'
    """
    def __init__(self, X_train, y_train, vectorizer_type='count', ngram_range=(1,1), alpha=1.0, class_prior=None):
        texts = [preprocess(x) for x in X_train]

        if vectorizer_type == 'tfidf':
            self.vectorizer = TfidfVectorizer(ngram_range=ngram_range)
        else:
            self.vectorizer = CountVectorizer(ngram_range=ngram_range)

        X_vec = self.vectorizer.fit_transform(texts)

        # class_prior should be list-like of length n_classes or None
        self.model = MultinomialNB(alpha=alpha, class_prior=class_prior)
        self.model.fit(X_vec, y_train)

    def predict(self, X):
        texts = [preprocess(x) for x in X]
        X_vec = self.vectorizer.transform(texts)
        return self.model.predict(X_vec)