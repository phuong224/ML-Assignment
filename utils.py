# src/utils.py
import nltk

# Ensure necessary NLTK data is available (first-run will download)
try:
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
except Exception:
    nltk.download("punkt")
    nltk.download("stopwords")
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer

def preprocess(sentence):
    """
    Hàm tiền xử lý một câu:
    - lowercase
    - tokenize
    - remove punctuation / non-alpha tokens
    - remove stopwords (English)
    - stemming (PorterStemmer)
    """
    if sentence is None:
        return ""
    sentence = str(sentence).lower()

    tokens = word_tokenize(sentence)
    words = [word for word in tokens if word.isalpha()]

    stop_words = set(stopwords.words('english'))
    words = [w for w in words if w not in stop_words]

    stemmer = PorterStemmer()
    stems = [stemmer.stem(w) for w in words]

    return " ".join(stems)
