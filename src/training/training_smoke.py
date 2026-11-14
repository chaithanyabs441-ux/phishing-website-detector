"""Tiny deterministic classical text-model fit for CI wiring checks."""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score


def run_smoke_training():
    train_urls = [
        "https://weather.example.org/today",
        "https://museum.example.net/visit",
        "https://library.example.edu/catalog",
        "https://parks.example.org/events",
        "https://secure-login-account-check.example.com/verify",
        "https://wallet-password-reset.example.net/auth",
        "https://bank-account-confirm.example.org/login",
        "https://payment-invoice-update.example.com/signin",
    ]
    train_labels = np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int8)
    valid_urls = [
        "https://gallery.example.net/hours",
        "https://account-security-verify.example.org/login",
        "https://community.example.edu/news",
        "https://wallet-secure-password.example.com/update",
    ]
    valid_labels = np.asarray([0, 1, 0, 1], dtype=np.int8)

    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), min_df=1)
    train_matrix = vectorizer.fit_transform(train_urls)
    valid_matrix = vectorizer.transform(valid_urls)
    model = LogisticRegression(solver="liblinear", random_state=42)
    model.fit(train_matrix, train_labels)
    scores = model.predict_proba(valid_matrix)[:, 1]
    ap = float(average_precision_score(valid_labels, scores))
    if len(scores) != len(valid_urls) or not np.isfinite(scores).all():
        raise AssertionError("smoke model emitted invalid validation scores")
    print("smoke_fit_rows=8 validation_rows=4 features={} AP={:.4f}".format(
        train_matrix.shape[1], ap,
    ))
    return ap


if __name__ == "__main__":
    run_smoke_training()
