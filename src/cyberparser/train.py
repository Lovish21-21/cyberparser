import argparse
import sklearn_crfsuite
from sklearn_crfsuite import metrics

def train_crf(x_train,y_train):
    crf=sklearn_crfsuite.CRF(
        algorithm="lbfgs",
        c1=0.1,          # L1 regularization
        c2=0.1,          # L2 regularization
        max_iterations=100,
        all_possible_transitions=True,

    )
    crf.fit(X_train, y_train)
    return crf

def evaluate_crf(crf, X_test, y_test):
    labels = list(crf.classes_)
    if "O" in labels:
        labels.remove("O")  # exclude "O" from F1 — inflates score otherwise
 
    y_pred = crf.predict(X_test)
 
    print("\n=== Classification Report ===")
    print(metrics.flat_classification_report(y_test, y_pred, labels=labels, digits=3))
 
    f1 = metrics.flat_f1_score(y_test, y_pred, average="weighted", labels=labels)
    print(f"Weighted F1 (excluding 'O'): {f1:.4f}")
    return y_pred