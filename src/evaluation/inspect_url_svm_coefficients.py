import joblib
import pandas as pd
from pathlib import Path


MODEL_PATH = Path("models")


def main():

    model = joblib.load(
        MODEL_PATH / "combined_linear_svm.joblib"
    )

    feature_names = [
        "url_length",
        "hostname_length",
        "path_length",
        "num_dots",
        "num_hyphens",
        "num_digits",
        "num_special_chars",
        "num_subdomains",
        "num_query_params",
        "has_at_symbol",
        "has_ip_address",
        "has_https",
        "suspicious_word_count"
    ]

    # The last 13 model coefficients
    # correspond to our hand-crafted URL features.
    coefficients = model.coef_[0][-13:]

    results = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coefficients
    })

    results["absolute_importance"] = (
        results["coefficient"].abs()
    )

    results = results.sort_values(
        "absolute_importance",
        ascending=False
    )

    print("\n" + "=" * 60)
    print("URL FEATURE IMPORTANCE")
    print("=" * 60)

    print(
        results[
            ["feature", "coefficient"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()