import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from src.evaluation.phase5_text import (
    MODEL_SPECS,
    fit_text_model,
    fit_text_models_shared,
    grouped_oof_text_scores,
    make_vectorizer,
)
from src.evaluation.evaluate_phase5_text import _paired_ap_bootstrap


def _text_frame(size=100):
    targets = np.tile([0, 1], size // 2)
    return pd.DataFrame({
        "target": targets,
        "group_domain": [f"domain-{index // 10}" for index in range(size)],
        "url": [
            f"https://site-{index // 10}.example/{'safe' if target == 0 else 'login'}/{index}"
            for index, target in enumerate(targets)
        ],
        "cleaned_text": ["welcome article" if target == 0 else "secure account login" for target in targets],
    })


def test_model_specs_cover_classical_url_and_cleaned_html_options():
    assert len(MODEL_SPECS) >= 10
    assert MODEL_SPECS["html_word_svc"][0] == "cleaned_html"
    assert MODEL_SPECS["hash_sgd_log_loss"][1] == "hashing_char"
    assert all("neural" not in spec[2] for spec in MODEL_SPECS.values())


def test_vectorizer_vocabulary_is_fitted_only_on_training_text():
    vectorizer = make_vectorizer("tfidf_char_2_5_min_df1")
    vectorizer.fit(["https://train-only.example/alpha"])

    assert "qzx" not in vectorizer.vocabulary_
    transformed = vectorizer.transform(["https://validation-only.example/qzxv-token"])
    assert transformed.shape[1] == len(vectorizer.vocabulary_)


def test_grouped_oof_scores_are_aligned_once_and_reproducible():
    train = _text_frame(100)
    validation = _text_frame(20).drop(columns=["target", "group_domain"])

    scores_a, validation_a, _, metadata = grouped_oof_text_scores(
        "char_svc_2_5", train, validation, folds=5,
    )
    scores_b, validation_b, _, _ = grouped_oof_text_scores(
        "char_svc_2_5", train, validation, folds=5,
    )

    assert len(scores_a) == len(train)
    assert len(validation_a) == len(validation)
    assert metadata["oof_assignment_min"] == metadata["oof_assignment_max"] == 1
    assert np.array_equal(scores_a, scores_b)
    assert np.array_equal(validation_a, validation_b)


def test_label_shuffle_performance_is_chance_level():
    rng = np.random.default_rng(42)
    urls = ["https://" + "".join(rng.choice(list("abcdef012345"), 24)) for _ in range(600)]
    shuffled_targets = rng.permutation(np.tile([0, 1], 300))
    train_x = make_vectorizer("tfidf_char_2_5_min_df1").fit_transform(urls[:400])
    validation_x = make_vectorizer("tfidf_char_2_5_min_df1").fit(urls[:400]).transform(urls[400:])
    from sklearn.svm import LinearSVC

    model = LinearSVC(class_weight="balanced", random_state=42, dual=True).fit(train_x, shuffled_targets[:400])
    score = average_precision_score(shuffled_targets[400:], model.decision_function(validation_x))

    assert 0.3 <= score <= 0.7


def test_text_model_roundtrip_and_single_batch_prediction_parity(tmp_path):
    train = _text_frame(100)
    validation = _text_frame(20).drop(columns=["target", "group_domain"])
    artifact = tmp_path / "text_model.joblib"
    vectorizer, model, batch_scores, _ = fit_text_model(
        "char_svc_2_5", train, validation, save_path=artifact,
    )
    restored = joblib.load(artifact)
    single = restored["model"].decision_function(
        restored["vectorizer"].transform([validation.url.iloc[0]])
    )

    assert np.allclose(batch_scores[:1], single)
    assert np.allclose(
        batch_scores,
        restored["model"].decision_function(vectorizer.transform(validation.url.astype(str))),
    )


def test_shared_vectorizer_fits_once_for_multiple_classifiers(tmp_path):
    train = _text_frame(100)
    validation = _text_frame(20).drop(columns=["target", "group_domain"])

    results = fit_text_models_shared(
        ["char_svc_2_5", "char_complement_nb"],
        train, validation, artifact_dir=tmp_path,
    )

    assert results["char_svc_2_5"]["metadata"]["vectorizer_shared_with"] == ["char_complement_nb"]
    assert results["char_complement_nb"]["metadata"]["vectorizer_shared_with"] == ["char_svc_2_5"]
    assert len(results["char_svc_2_5"]["scores"]) == len(validation)
    assert len(results["char_complement_nb"]["scores"]) == len(validation)


def test_paired_domain_bootstrap_is_zero_for_identical_predictions():
    targets = np.tile([0, 1], 100)
    scores = np.linspace(-2, 2, len(targets))
    domain_codes = np.repeat(np.arange(20), 10)

    result = _paired_ap_bootstrap(
        targets, scores, scores, domain_codes, 20, replicates=30, seed=5,
    )

    assert result["point_difference"] == 0.0
    assert result["ci_95"] == [0.0, 0.0]
    assert result["replicates_completed"] == 30