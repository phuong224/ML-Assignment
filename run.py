# run.py
import os
import joblib
from experiments import run_all_experiments, gridsearch_tfidf_nb
from data_loader import load_dataset
from models import NaiveBayesModel
from sklearn.metrics import confusion_matrix, classification_report

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def save_final_model():
    """
    Lưu mô hình cuối cùng: MultinomialNB + TF-IDF + best ngram + balanced prior
    Các tham số `best_ngram` và `best_alpha` được chọn dựa trên kết quả
    từ quá trình chạy thử nghiệm trong `experiments.py`.
    """
    X, y = load_dataset()

    gs = gridsearch_tfidf_nb()
    best_ngram = gs['best_params']['vectorizer__ngram_range']
    best_alpha = gs['best_params']['classifier__alpha']

    print (f"best ngram: {best_ngram}, best alpha: {best_alpha}")

    n_classes = len(y.unique())
    class_prior = [1.0 / n_classes] * n_classes

    model = NaiveBayesModel(
        X, y,
        vectorizer_type='tfidf',
        ngram_range=best_ngram,
        alpha=best_alpha,
        class_prior=class_prior
    )

    # Lưu model
    model_path = os.path.join(MODEL_DIR, "final_nb_model.joblib")
    joblib.dump(model, model_path)
    print(f"Final model saved to {model_path}")

def test_model(model_path):
    nb_model = joblib.load(model_path)

    X, y = load_dataset()
    y_pred = nb_model.predict(X)

    # Confusion Matrix
    cm = confusion_matrix(y, y_pred, labels=['ham', 'smishing', 'spam'])
    print(cm)

    # Classification Report (precision, recall, f1)
    print(classification_report(y, y_pred, labels=['ham', 'smishing', 'spam']))

def main():
    # Chạy tất cả experiments và lưu CSV
    run_all_experiments()
    # Lưu mô hình cuối cùng
    save_final_model()
    # in confusion matrix
    test_model(os.path.join(MODEL_DIR, "final_nb_model.joblib"))


if __name__ == "__main__":
    main()
