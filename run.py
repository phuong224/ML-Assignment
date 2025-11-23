# run.py
import os
import joblib
from experiments import run_all_experiments
from data_loader import load_dataset
from models import NaiveBayesModel

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def save_final_model(best_ngram, best_alpha):
    """
    Lưu mô hình cuối cùng: MultinomialNB + TF-IDF + best ngram + balanced prior
    """
    X, y = load_dataset()

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

def main():
    # Chạy tất cả experiments và lưu CSV
    best_params = run_all_experiments()
    # Lưu mô hình cuối cùng
    save_final_model(best_params['ngram_range'], best_params['alpha'])

if __name__ == "__main__":
    main()
