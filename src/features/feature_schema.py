"""Shared feature-column schemas used by training, evaluation, and serving."""

from src.features.html_features_v2 import HTML_FEATURE_COLUMNS_V2
from src.features.url_features import URL_FEATURE_COLUMNS_V2


NUMERIC_HTML_FEATURES = [
    column for column in HTML_FEATURE_COLUMNS_V2 if column != "cleaned_text"
]
PHASE4_FEATURE_COLUMNS = [*URL_FEATURE_COLUMNS_V2, *NUMERIC_HTML_FEATURES]
CATEGORICAL_FEATURES = ["registered_domain", "public_suffix", "tld_category"]
