# src/models.py
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
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

class SVMModel:
    """
    Simple LinearSVC wrapper with TF-IDF (1,2) by default.
    Uses class_weight='balanced' by default.
    """
    def __init__(self, X_train, y_train, ngram_range=(1,2)):
        texts = [preprocess(x) for x in X_train]
        self.vectorizer = TfidfVectorizer(ngram_range=ngram_range)
        X_vec = self.vectorizer.fit_transform(texts)

        self.model = LinearSVC(class_weight="balanced", max_iter=20000)
        self.model.fit(X_vec, y_train)

    def predict(self, X):
        texts = [preprocess(x) for x in X]
        X_vec = self.vectorizer.transform(texts)
        return self.model.predict(X_vec)
