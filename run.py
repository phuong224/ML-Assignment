# run.py
import os
import joblib
from experiments import run_all_experiments
from data_loader import load_dataset
from models import NaiveBayesModel

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def save_final_model():
    """
    Lưu mô hình cuối cùng: MultinomialNB + TF-IDF + best ngram + balanced prior
    Các tham số `best_ngram` và `best_alpha` được chọn dựa trên kết quả
    từ quá trình chạy thử nghiệm trong `experiments.py`.
    """
    X, y = load_dataset()

    best_ngram = (1, 2) 
    best_alpha = 0.1   

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
    run_all_experiments()
    # Lưu mô hình cuối cùng
    save_final_model()

if __name__ == "__main__":
    main()
