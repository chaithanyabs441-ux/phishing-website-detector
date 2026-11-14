"""Shared scoring metrics and registered-domain bootstrap confidence intervals."""

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


PREVALENCES = (0.0005, 0.005, 0.01, 0.05, 0.45)


def _metric_bundle(y_true, scores, weights=None):
    y_true = np.asarray(y_true, dtype=np.int8)
    scores = np.asarray(scores)
    weights = np.ones(len(y_true), dtype=np.float64) if weights is None else np.asarray(weights)
    predicted = scores >= 0.0
    positive = y_true == 1
    negative = ~positive
    tp = float(weights[predicted & positive].sum())
    fp = float(weights[predicted & negative].sum())
    fn = float(weights[~predicted & positive].sum())
    tn = float(weights[~predicted & negative].sum())
    tpr = tp / max(1.0, tp + fn)
    fpr = fp / max(1.0, fp + tn)
    precision = tp / max(1.0, tp + fp)
    recall = tpr
    f1 = 2 * precision * recall / max(1e-15, precision + recall)
    total = tp + fp + fn + tn
    metrics = {
        "average_precision": float(average_precision_score(y_true, scores, sample_weight=weights)),
        "roc_auc": float(roc_auc_score(y_true, scores, sample_weight=weights)),
        "accuracy": (tp + tn) / max(1.0, total),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": fpr,
        "true_positive_rate": tpr,
    }
    for prevalence in PREVALENCES:
        alerts = prevalence * tpr + (1.0 - prevalence) * fpr
        precision_at_prevalence = prevalence * tpr / max(1e-15, alerts)
        label = f"{prevalence * 100:g}%"
        metrics[f"precision_at_prevalence_{label}"] = precision_at_prevalence
        metrics[f"alerts_per_1000_at_prevalence_{label}"] = 1000.0 * alerts
    return metrics


def _domain_bootstrap(y_true, scores, domain_codes, n_domains, replicates, seed):
    rng = np.random.default_rng(seed)
    results = []
    for _ in range(replicates):
        sampled = rng.integers(0, n_domains, size=n_domains)
        multiplicity = np.bincount(sampled, minlength=n_domains)
        weights = multiplicity[domain_codes]
        if len(np.unique(y_true[weights > 0])) < 2:
            continue
        results.append(_metric_bundle(y_true, scores, weights))
    keys = results[0].keys()
    return {
        key: [float(np.quantile([row[key] for row in results], 0.025)),
              float(np.quantile([row[key] for row in results], 0.975))]
        for key in keys
    }, len(results)
