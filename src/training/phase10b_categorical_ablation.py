"""Compare Phase 4 CatBoost categorical feature variants on temporal validation."""

import json
import time

import numpy as np
import pandas as pd
import yaml
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from src.data.engineered_feature_dataset import (
    CATEGORICAL_FEATURES,
    PHASE4_FEATURE_COLUMNS,
    PROJECT_ROOT,
    load_phase4_data,
)


RESULTS_PATH = PROJECT_ROOT / "experiments" / "phase10b_categorical_ablation.json"
REPORT_PATH = PROJECT_ROOT / "reports" / "phase_10b_categorical_ablation.md"
PHASE4_RESULTS_PATH = PROJECT_ROOT / "experiments" / "phase4_gbm_results.json"
CONFIG_PATH = PROJECT_ROOT / "configs" / "phase4_gbm.yaml"
BOOTSTRAP_REPLICATES = 200
BOOTSTRAP_SEED = 42
NONINFERIORITY_MARGIN = 0.01

VARIANTS = {
    "A": {
        "description": "Current full feature set",
        "categorical": ["registered_domain", "public_suffix", "tld_category"],
        "drop": [],
    },
    "B": {
        "description": "Public suffix and TLD category categorical; registered domain excluded",
        "categorical": ["public_suffix", "tld_category"],
        "drop": ["registered_domain"],
    },
    "C": {
        "description": "Numeric features only; all categorical features excluded",
        "categorical": [],
        "drop": list(CATEGORICAL_FEATURES),
    },
}


def prepare_variant_frames(train_frame, validation_frame, categorical, drop):
    """Fit categorical vocabularies and numeric medians on train rows only."""
    columns = [column for column in PHASE4_FEATURE_COLUMNS if column not in drop]
    train_features = train_frame[columns].copy()
    validation_features = validation_frame[columns].copy()
    for column in categorical:
        train_values = train_features[column].fillna("__MISSING__").astype(str)
        categories = sorted(set(train_values) | {"__UNKNOWN__"})
        validation_values = validation_features[column].fillna("__MISSING__").astype(str)
        validation_values = validation_values.where(
            validation_values.isin(categories), "__UNKNOWN__",
        )
        train_features[column] = pd.Categorical(train_values, categories=categories)
        validation_features[column] = pd.Categorical(
            validation_values, categories=categories,
        )

    numeric = [column for column in columns if column not in categorical]
    medians = train_features[numeric].median()
    train_features[numeric] = train_features[numeric].fillna(medians).fillna(0)
    validation_features[numeric] = validation_features[numeric].fillna(medians).fillna(0)
    return train_features, validation_features


def _bootstrap_differences(target, predictions, domains, replicates, seed):
    """Compute paired registered-domain bootstrap intervals against A."""
    target = np.asarray(target, dtype=np.int8)
    domain_values = np.asarray(domains, dtype=str)
    unique_domains, domain_codes = np.unique(domain_values, return_inverse=True)
    rng = np.random.default_rng(seed)
    differences = {name: {"average_precision": [], "roc_auc": []} for name in ("B", "C")}
    absolute = {name: {"average_precision": [], "roc_auc": []} for name in VARIANTS}

    for _ in range(replicates):
        sampled = rng.integers(0, len(unique_domains), size=len(unique_domains))
        weights = np.bincount(sampled, minlength=len(unique_domains))[domain_codes]
        included = weights > 0
        if np.unique(target[included]).size != 2:
            continue
        current = {}
        for name, score in predictions.items():
            score = np.asarray(score, dtype=np.float64)
            current[name] = {
                "average_precision": float(
                    average_precision_score(target, score, sample_weight=weights)
                ),
                "roc_auc": float(roc_auc_score(target, score, sample_weight=weights)),
            }
            for metric in current[name]:
                absolute[name][metric].append(current[name][metric])
        for name in ("B", "C"):
            for metric in current[name]:
                differences[name][metric].append(
                    current[name][metric] - current["A"][metric]
                )

    def quantiles(values):
        return [
            float(np.quantile(values, 0.025)),
            float(np.quantile(values, 0.975)),
        ]
    return {
        "completed_replicates": len(differences["B"]["average_precision"]),
        "absolute_95ci": {
            name: {metric: quantiles(values) for metric, values in metrics.items()}
            for name, metrics in absolute.items()
        },
        "paired_difference_95ci_vs_A": {
            name: {metric: quantiles(values) for metric, values in metrics.items()}
            for name, metrics in differences.items()
        },
    }


def _fit_variant(name, spec, train_frame, validation_frame, params, seed, n_jobs):
    x_train, x_validation = prepare_variant_frames(
        train_frame,
        validation_frame,
        spec["categorical"],
        spec["drop"],
    )
    y_train = train_frame["target"].to_numpy(dtype=np.int8)
    y_validation = validation_frame["target"].to_numpy(dtype=np.int8)
    model = CatBoostClassifier(
        **params,
        iterations=3000,
        loss_function="Logloss",
        eval_metric="PRAUC",
        auto_class_weights="Balanced",
        random_seed=seed,
        thread_count=n_jobs,
        allow_writing_files=False,
        verbose=False,
    )
    started = time.monotonic()
    model.fit(
        x_train,
        y_train,
        eval_set=(x_validation, y_validation),
        cat_features=spec["categorical"],
        early_stopping_rounds=100,
        use_best_model=True,
        verbose=False,
    )
    elapsed = time.monotonic() - started
    probabilities = model.predict_proba(x_validation)[:, 1]
    return {
        "probabilities": probabilities,
        "metrics": {
            "average_precision": float(average_precision_score(y_validation, probabilities)),
            "roc_auc": float(roc_auc_score(y_validation, probabilities)),
        },
        "training_seconds": elapsed,
        "best_iteration": int(model.best_iteration_),
        "feature_count": int(x_train.shape[1]),
        "categorical_features": spec["categorical"],
        "dropped_features": spec["drop"],
    }


def select_variant(results, bootstrap):
    """Apply the preregistered paired AP noninferiority rule."""
    ci = bootstrap["paired_difference_95ci_vs_A"]
    eligible = [
        name for name in ("B", "C")
        if ci[name]["average_precision"][0] >= -NONINFERIORITY_MARGIN
    ]
    if not eligible:
        return "A", eligible
    eligible.sort(key=lambda name: (results[name]["metrics"]["average_precision"],
                                    -len(VARIANTS[name]["categorical"])), reverse=True)
    chosen = eligible[0]
    if len(eligible) == 2 and abs(
        results[eligible[0]]["metrics"]["average_precision"]
        - results[eligible[1]]["metrics"]["average_precision"]
    ) <= 0.0001:
        chosen = min(eligible, key=lambda name: len(VARIANTS[name]["categorical"]))
    return chosen, eligible


def run_ablation():
    """Fit A/B/C against temporal validation only and save its record."""
    started = time.monotonic()
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    phase4 = json.loads(PHASE4_RESULTS_PATH.read_text(encoding="utf-8"))
    params = phase4["tuning"]["catboost"]["best_params"]
    data = load_phase4_data()
    train = data.loc[data["split"] == "train"].reset_index(drop=True)
    validation = data.loc[data["split"] == "validation"].reset_index(drop=True)
    if train.empty or validation.empty:
        raise ValueError("Frozen temporal train and validation splits must be nonempty")

    seed = int(config["seed"])
    n_jobs = int(config["tuning"]["n_jobs"])
    y_validation = validation["target"].to_numpy(dtype=np.int8)
    train_domains = set(train["group_domain"].fillna("").astype(str))
    validation_domains = validation["group_domain"].fillna("").astype(str)
    unseen_mask = ~validation_domains.isin(train_domains).to_numpy()
    if np.unique(y_validation[unseen_mask]).size != 2:
        raise ValueError("Temporal validation unseen-domain rows must contain both classes")

    results = {}
    predictions = {}
    for name, spec in VARIANTS.items():
        result = _fit_variant(name, spec, train, validation, params, seed, n_jobs)
        predictions[name] = result.pop("probabilities")
        result["validation_unseen_domain_average_precision"] = float(
            average_precision_score(y_validation[unseen_mask], predictions[name][unseen_mask])
        )
        results[name] = result
        print(
            f"{name}: AP={result['metrics']['average_precision']:.6f}; "
            f"ROC-AUC={result['metrics']['roc_auc']:.6f}; "
            f"AP_unseen={result['validation_unseen_domain_average_precision']:.6f}; "
            f"best_iteration={result['best_iteration']}; "
            f"seconds={result['training_seconds']:.1f}",
            flush=True,
        )

    bootstrap = _bootstrap_differences(
        y_validation,
        predictions,
        validation_domains,
        int(config["bootstrap"]["replicates"]),
        seed,
    )
    chosen, eligible = select_variant(results, bootstrap)
    output = {
        "experiment": "phase10b_temporal_catboost_categorical_ablation",
        "split": "frozen temporal train/validation only",
        "training_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "validation_domains": int(validation_domains.nunique()),
        "validation_unseen_domain_rows": int(unseen_mask.sum()),
        "validation_unseen_domain_count": int(validation_domains[unseen_mask].nunique()),
        "params_source": "experiments/phase4_gbm_results.json tuning.catboost.best_params",
        "params": params,
        "early_stopping_rounds": 100,
        "bootstrap": {
            "resampling_unit": "registered_domain",
            "replicates": bootstrap["completed_replicates"],
            "seed": seed,
            "confidence_interval": "paired percentile 95%",
        },
        "variants": results,
        "bootstrap_results": bootstrap,
        "decision_rule": {
            "noninferiority_margin_ap": NONINFERIORITY_MARGIN,
            "eligible_variants": eligible,
            "selected_variant": chosen,
            "selection_metric": "full temporal-validation AP among candidates whose paired CI lower bound is >= -0.01",
        },
        "official_test_accessed": False,
        "wall_seconds": time.monotonic() - started,
    }
    RESULTS_PATH.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    _append_report(output)
    print(f"Selected feature variant: {chosen}")
    print(f"Wrote experiment record: {RESULTS_PATH}")
    print(f"Wrote report: {REPORT_PATH}")
    return output


def _append_report(output):
    existing = REPORT_PATH.read_text(encoding="utf-8")
    placeholder = "Results will be appended after the preregistered rule by `python -m src.evaluation.phase10b_categorical_ablation`."
    if placeholder not in existing:
        raise ValueError("Expected preregistered report placeholder was not found")
    rows = []
    for name, result in output["variants"].items():
        ap_ci = output["bootstrap_results"]["absolute_95ci"][name]["average_precision"]
        auc_ci = output["bootstrap_results"]["absolute_95ci"][name]["roc_auc"]
        if name == "A":
            ap_delta = auc_delta = "reference"
        else:
            ap_delta_ci = output["bootstrap_results"]["paired_difference_95ci_vs_A"][name]
            ap_delta = (
                f"{result['metrics']['average_precision'] - output['variants']['A']['metrics']['average_precision']:+.6f} "
                f"[{ap_delta_ci['average_precision'][0]:+.6f}, {ap_delta_ci['average_precision'][1]:+.6f}]"
            )
            auc_delta = (
                f"{result['metrics']['roc_auc'] - output['variants']['A']['metrics']['roc_auc']:+.6f} "
                f"[{ap_delta_ci['roc_auc'][0]:+.6f}, {ap_delta_ci['roc_auc'][1]:+.6f}]"
            )
        rows.append(
            f"| {name} | {result['feature_count']} | {result['best_iteration']} | "
            f"{result['metrics']['average_precision']:.6f} [{ap_ci[0]:.6f}, {ap_ci[1]:.6f}] | "
            f"{result['metrics']['roc_auc']:.6f} [{auc_ci[0]:.6f}, {auc_ci[1]:.6f}] | "
            f"{ap_delta} | {auc_delta} | "
            f"{result['validation_unseen_domain_average_precision']:.6f} |"
        )
    details = f"""### Execution results

The experiment used {output['training_rows']:,} temporal training rows and {output['validation_rows']:,} temporal validation rows. The validation set had {output['validation_domains']:,} registered domains; {output['validation_unseen_domain_rows']:,} rows across {output['validation_unseen_domain_count']:,} domains were absent from the training partition. Unseen-domain AP is computed on those validation rows only.

| Variant | Features | Best iteration | AP (95% domain-bootstrap CI) | ROC-AUC (95% domain-bootstrap CI) | Paired ΔAP vs A (95% CI) | Paired ΔROC-AUC vs A (95% CI) | AP, unseen validation domains |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

- Registered-domain bootstrap: {output['bootstrap']['replicates']} paired replicates, seed {output['bootstrap']['seed']}.
- Candidate variants eligible under the preregistered AP rule: {', '.join(output['decision_rule']['eligible_variants']) or 'none'}.
- Decision: **{output['decision_rule']['selected_variant']}**.
- Official test data accessed: **no**.
- Experiment record: `{RESULTS_PATH.relative_to(PROJECT_ROOT).as_posix()}`.

The variant choice is based only on full temporal-validation AP and the paired confidence interval rule above. The unseen-domain metric is descriptive. No official-test outputs or benchmarks were used.
"""
    REPORT_PATH.write_text(existing.replace(placeholder, details), encoding="utf-8")


def main():
    run_ablation()


if __name__ == "__main__":
    main()
