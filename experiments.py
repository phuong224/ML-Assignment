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

def run_kfold_experiment(model_class, vectorizer_type='tfidf', ngram_range=(1,1),
                         alpha=1.0, class_prior=None, n_splits=5, save_csv=None):
    X, y = load_dataset()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    rows = []
    accs, f1s = [], []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        if model_class == NaiveBayesModel:
            model = model_class(X_train, y_train,
                        vectorizer_type=vectorizer_type,
                        ngram_range=ngram_range,
                        alpha=alpha,
                        class_prior=class_prior)
        elif model_class == SVMModel:
            model = model_class(X_train, y_train,
                        ngram_range=ngram_range)
        else:
            raise ValueError(f"Unsupported model_class: {model_class}")



        y_pred = model.predict(X_test)
        acc_fold = accuracy_score(y_test, y_pred)
        f1_fold = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        accs.append(acc_fold)
        f1s.append(f1_fold)

        row = {
            'fold': fold_idx,
            'accuracy': acc_fold,
            'f1_weighted': f1_fold,
            'model': model_class.__name__,
            'vectorizer': vectorizer_type,
            'ngram_range': ngram_range,
            'alpha': alpha,
            'class_prior': class_prior
        }
        rows.append(row)

    summary = {
        'accuracy_mean': np.mean(accs),
        'accuracy_std': np.std(accs),
        'f1_mean': np.mean(f1s),
        'f1_std': np.std(f1s)
    }

    if save_csv:
        save_path = os.path.join(TRAINING_DIR, save_csv)
        df_rows = pd.DataFrame(rows)
        summary_row = {**summary, 'fold': 'summary'}
        df_rows = pd.concat([df_rows, pd.DataFrame([summary_row])], ignore_index=True)
        df_rows.to_csv(save_path, index=False)

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
    # 1. Baseline CountVectorizer
    run_kfold_experiment(NaiveBayesModel,
                         vectorizer_type='count',
                         ngram_range=(1,1),
                         alpha=1.0,
                         save_csv="results_baseline_count.csv")

    # 2. TF-IDF unigram
    run_kfold_experiment(NaiveBayesModel,
                         vectorizer_type='tfidf',
                         ngram_range=(1,1),
                         alpha=1.0,
                         save_csv="results_tfidf_uni.csv")

    # 3. GridSearch
    gs = gridsearch_tfidf_nb()
    best_ngram = gs['best_params']['vectorizer__ngram_range']
    best_alpha = gs['best_params']['classifier__alpha']
    run_kfold_experiment(NaiveBayesModel,
                         vectorizer_type='tfidf',
                         ngram_range=best_ngram,
                         alpha=best_alpha,
                         save_csv="results_gridsearch_nb.csv")

    # 4. Balanced prior
    n_classes = len(load_dataset()[1].unique())
    class_prior = [1.0 / n_classes] * n_classes
    run_kfold_experiment(NaiveBayesModel,
                         vectorizer_type='tfidf',
                         ngram_range=best_ngram,
                         alpha=best_alpha,
                         class_prior=class_prior,
                         save_csv="results_balanced_prior.csv")

    # 5. Linear SVM
    run_kfold_experiment(SVMModel,
                         vectorizer_type='tfidf',
                         ngram_range=best_ngram,
                         save_csv="results_svm.csv")

    print(f"All experiments completed. CSV files saved in '{TRAINING_DIR}' folder.")

    return {
        'ngram_range': best_ngram,
        'alpha': best_alpha
    }
