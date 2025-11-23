# src/experiments.py
import os
import numpy as np
import pandas as pd
from data_loader import load_dataset
from models import NaiveBayesModel, SVMModel
from utils import preprocess
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, f1_score, accuracy_score

TRAINING_DIR = "training_results"
os.makedirs(TRAINING_DIR, exist_ok=True)

def run_kfold_experiment(X, y, model_class, vectorizer_type='tfidf', ngram_range=(1,1),
                         alpha=1.0, class_prior=None, n_splits=5, save_csv=None):

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    rows = []
    accs, f1s, macro_f1s = [], [], []

    class_labels = sorted(y.unique())
    per_class_metrics = {label: {'precision': [], 'recall': [], 'f1-score': []}
                         for label in class_labels}

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Khởi tạo model
        if model_class == NaiveBayesModel:
            model = model_class(
                X_train, y_train,
                vectorizer_type=vectorizer_type,
                ngram_range=ngram_range,
                alpha=alpha,
                class_prior=class_prior
            )
        elif model_class == SVMModel:
            model = model_class(X_train, y_train, ngram_range=ngram_range)
        else:
            raise ValueError(f"Unsupported model_class: {model_class}")

        y_pred = model.predict(X_test)

        # === METRICS CHÍNH ===
        acc_fold = accuracy_score(y_test, y_pred)
        f1_fold = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        macro_p = report['macro avg']['precision']
        macro_r = report['macro avg']['recall']
        macro_f1 = report['macro avg']['f1-score']

        # Lưu per-class
        for label in class_labels:
            str_label = str(label)
            if str_label in report:
                for metric in ['precision', 'recall', 'f1-score']:
                    per_class_metrics[label][metric].append(report[str_label][metric])

        # Lưu tổng hợp
        accs.append(acc_fold)
        f1s.append(f1_fold)
        macro_f1s.append(macro_f1)

        rows.append({
            'fold': fold_idx,
            'accuracy': acc_fold,
            'f1_weighted': f1_fold,
            'macro_precision': macro_p,
            'macro_recall': macro_r,
            'macro_f1': macro_f1
        })

    # === SUMMARY ===
    summary = {
        'accuracy_mean': np.mean(accs),
        'accuracy_std': np.std(accs),
        'f1_weighted_mean': np.mean(f1s),
        'f1_weighted_std': np.std(f1s),
        'macro_f1_mean': np.mean(macro_f1s),
        'macro_f1_std': np.std(macro_f1s)
    }

    print("\n" + "="*25 + " OVERALL SUMMARY " + "="*25)
    print(f"Model        : {model_class.__name__}")
    print(f"Vectorizer   : {vectorizer_type}, ngram_range: {ngram_range}")
    if model_class == NaiveBayesModel:
        print(f"Alpha        : {alpha}, class_prior: {class_prior}")
    print("-" * 67)
    print(f"Accuracy     : {summary['accuracy_mean']:.4f} (+/- {summary['accuracy_std']:.4f})")
    print(f"F1-Weighted  : {summary['f1_weighted_mean']:.4f} (+/- {summary['f1_weighted_std']:.4f})")
    print(f"Macro F1     : {summary['macro_f1_mean']:.4f} (+/- {summary['macro_f1_std']:.4f})")

    # === Per-Class Summary ===
    print("\n--- Per-Class Average Metrics Across All Folds ---")
    for label in class_labels:
        print(f"Class {label}:")
        for metric in ['precision', 'recall', 'f1-score']:
            avg = np.mean(per_class_metrics[label][metric])
            std = np.std(per_class_metrics[label][metric])
            print(f"   {metric}: {avg:.4f} (+/- {std:.4f})")
    print("="*67 + "\n")

    # === Lưu CSV ===
    if save_csv:
        df_rows = pd.DataFrame(rows)
        df_rows = df_rows.dropna(axis=1, how='all')  # để chắc chắn không có cột trống
        df_rows.to_csv(os.path.join(TRAINING_DIR, save_csv), index=False)

    return summary




def gridsearch_tfidf_nb(n_splits=5, n_jobs=-1, verbose=1):
    X, y = load_dataset()
    X_prep = X.apply(preprocess)

    pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer()),
        ('classifier', MultinomialNB())
    ])

    param_grid = {
        'vectorizer__ngram_range': [(1,1), (1,2), (1,3)],
        'classifier__alpha': [0.01, 0.1, 0.5, 1.0]
    }

    grid = GridSearchCV(pipeline, param_grid, cv=n_splits, scoring='f1_weighted',
                        n_jobs=n_jobs, verbose=verbose)
    grid.fit(X_prep, y)

    return {'best_params': grid.best_params_, 'best_score': grid.best_score_}


def run_all_experiments():
    # Tải dữ liệu một lần duy nhất
    X, y = load_dataset()

    # 1. Baseline CountVectorizer
    run_kfold_experiment(X, y, NaiveBayesModel,
                         vectorizer_type='count',
                         ngram_range=(1,1),
                         alpha=1.0,
                         save_csv="results_baseline_count.csv")

    # 2. TF-IDF unigram
    run_kfold_experiment(X, y, NaiveBayesModel,
                         vectorizer_type='tfidf',
                         ngram_range=(1,1),
                         alpha=1.0,
                         save_csv="results_tfidf_uni.csv")

    # 3. GridSearch
    print("\n" + "="*25 + " RUNNING GRIDSEARCH " + "="*25)
    gs = gridsearch_tfidf_nb()
    best_ngram = gs['best_params']['vectorizer__ngram_range']
    best_alpha = gs['best_params']['classifier__alpha']
    print(f"GridSearch found best params: ngram_range={best_ngram}, alpha={best_alpha}")
    print("="*70 + "\n")

    run_kfold_experiment(X, y, NaiveBayesModel,
                         vectorizer_type='tfidf',
                         ngram_range=best_ngram,
                         alpha=best_alpha,
                         save_csv="results_gridsearch_nb.csv")

    # 4. Balanced prior
    n_classes = len(y.unique())
    class_prior = [1.0 / n_classes] * n_classes
    run_kfold_experiment(X, y, NaiveBayesModel,
                         vectorizer_type='tfidf',
                         ngram_range=best_ngram,
                         alpha=best_alpha,
                         class_prior=class_prior,
                         save_csv="results_balanced_prior.csv")

    # 5. Linear SVM
    run_kfold_experiment(X, y, SVMModel,
                         vectorizer_type='tfidf',
                         ngram_range=best_ngram,
                         save_csv="results_svm.csv")

    print(f"All experiments completed. CSV files saved in '{TRAINING_DIR}' folder.")

    return {
        'ngram_range': best_ngram,
        'alpha': best_alpha
    }
