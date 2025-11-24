# run.py
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from experiments import run_all_experiments, gridsearch_countvec_nb
from data_loader import load_dataset
from models import NaiveBayesModel
from sklearn.metrics import confusion_matrix, classification_report

MODEL_DIR = "models"
FIGURES_DIR = "figures"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


def save_final_model():
    """
    Lưu mô hình cuối cùng: MultinomialNB + CountVectorizer + best ngram.
    Các tham số `best_ngram` và `best_alpha` được chọn dựa trên kết quả
    từ quá trình chạy thử nghiệm trong `experiments.py`.
    """
    X, y = load_dataset()

    gs = gridsearch_countvec_nb()
    best_ngram = gs['best_params']['vectorizer__ngram_range']
    best_alpha = gs['best_params']['classifier__alpha']

    print (f"best ngram: {best_ngram}, best alpha: {best_alpha}")

    model = NaiveBayesModel(
        X, y,
        vectorizer_type='count',
        ngram_range=best_ngram,
        alpha=best_alpha,
        class_prior=None  # Sử dụng prior mặc định vì cho kết quả tốt hơn
    )

    # Lưu model
    model_path = os.path.join(MODEL_DIR, "final_nb_model.joblib")
    joblib.dump(model, model_path)
    print(f"Final model saved to {model_path}")

def test_model(model_path):
    nb_model = joblib.load(model_path)

    X, y = load_dataset()
    y_pred = nb_model.predict(X)

    labels = ['ham', 'smishing', 'spam']

    # Confusion Matrix
    cm = confusion_matrix(y, y_pred, labels=labels)
    print("Confusion Matrix:")
    print(cm)

    # Vẽ và lưu confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix for Final Model')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    figure_path = os.path.join(FIGURES_DIR, "final_model_confusion_matrix.png")
    plt.savefig(figure_path)
    print(f"Confusion matrix figure saved to {figure_path}")

    # Classification Report (precision, recall, f1)
    print("\nClassification Report:")
    print(classification_report(y, y_pred, labels=labels))

def main():
    # Chạy tất cả experiments và lưu CSV
    run_all_experiments()
    # Lưu mô hình cuối cùng
    save_final_model()
    # in confusion matrix
    test_model(os.path.join(MODEL_DIR, "final_nb_model.joblib"))


if __name__ == "__main__":
    main()
