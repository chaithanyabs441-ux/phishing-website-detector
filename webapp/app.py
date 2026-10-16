import os
import sys
import re

import joblib
import pandas as pd

from scipy.sparse import hstack, csr_matrix

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool


# ============================================================
# PATH SETUP
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

sys.path.append(PROJECT_ROOT)

MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "models"
)

GOOGLE_SAFE_BROWSING_API_KEY = os.environ.get(
    "GOOGLE_SAFE_BROWSING_API_KEY",
    "",
).strip()


# ============================================================
# CLASSIFICATION CONFIGURATION
# ============================================================

# ------------------------------------------------------------
# MODEL-SPECIFIC LINEAR SVM DECISION THRESHOLDS
# ------------------------------------------------------------
#
# URL TF-IDF's threshold is raised to suppress weak positive
# scores like the reported false positive. The other models
# retain their existing decision boundary because their scores
# are not directly comparable and no matching threshold report
# is available for them.
# ------------------------------------------------------------

CLASSIFICATION_THRESHOLDS = {
    "url_tfidf": 1.5,
    "url_engineered": 0.0,
    "url_html": 0.0,
}


# ============================================================
# FEATURE EXTRACTORS
# ============================================================

from src.features.url_features import extract_url_features
from src.features.html_features import extract_html_features
from src.reputation.google_safe_browsing import lookup_url as lookup_safe_browsing


# ============================================================
# LOAD MODELS
# ============================================================

print("=" * 60)
print("PHISH//X - Loading Models")
print("=" * 60)


# ------------------------------------------------------------
# MODEL 1
# URL TF-IDF
# ------------------------------------------------------------

url_tfidf = joblib.load(
    os.path.join(
        MODEL_DIR,
        "url_tfidf.joblib"
    )
)

url_svm = joblib.load(
    os.path.join(
        MODEL_DIR,
        "url_linear_svm.joblib"
    )
)

print("[OK] URL TF-IDF model loaded")


# ------------------------------------------------------------
# MODEL 2
# URL + ENGINEERED FEATURES
# ------------------------------------------------------------

combined_tfidf = joblib.load(
    os.path.join(
        MODEL_DIR,
        "combined_tfidf.joblib"
    )
)

combined_svm = joblib.load(
    os.path.join(
        MODEL_DIR,
        "combined_linear_svm.joblib"
    )
)

url_scaler = joblib.load(
    os.path.join(
        MODEL_DIR,
        "url_engineered_scaler.joblib"
    )
)

print("[OK] URL + Engineered Features model loaded")


# ------------------------------------------------------------
# MODEL 3
# URL + HTML
# ------------------------------------------------------------

url_html_tfidf = joblib.load(
    os.path.join(
        MODEL_DIR,
        "url_html_tfidf.joblib"
    )
)

url_html_svm = joblib.load(
    os.path.join(
        MODEL_DIR,
        "url_html_linear_svm.joblib"
    )
)

html_scaler = joblib.load(
    os.path.join(
        MODEL_DIR,
        "html_scaler.joblib"
    )
)

html_feature_columns = joblib.load(
    os.path.join(
        MODEL_DIR,
        "html_feature_columns.joblib"
    )
)

print("[OK] URL + HTML model loaded")

print("=" * 60)
for model_choice, threshold in CLASSIFICATION_THRESHOLDS.items():
    print("Classification threshold ({}) = {:.2f}".format(
        model_choice,
        threshold,
    ))
print("=" * 60)


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="PHISH//X",
    description="Explainable phishing website detection",
    version="1.0"
)


# ============================================================
# TEMPLATES
# ============================================================

templates = Jinja2Templates(
    directory=os.path.join(
        os.path.dirname(__file__),
        "templates"
    )
)


# ============================================================
# STATIC FILES
# ============================================================

app.mount(
    "/static",
    StaticFiles(
        directory=os.path.join(
            os.path.dirname(__file__),
            "static"
        )
    ),
    name="static"
)


# ============================================================
# URL FEATURE COLUMNS
# ============================================================

URL_FEATURE_COLUMNS = [
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


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def normalize_url(url):

    url = url.strip()

    if not url:
        return ""

    if not re.match(
        r"^https?://",
        url,
        re.IGNORECASE
    ):
        url = "https://" + url

    return url


# ============================================================
# MODEL SIGNAL INDICATOR
# ============================================================
#
# IMPORTANT:
#
# LinearSVC.decision_function() is NOT a calibrated
# probability.
# The UI reports score bands instead of a probability-like
# sigmoid value.
#
# ============================================================

def get_risk_level(decision_score, threshold):

    if decision_score >= threshold:
        return "HIGH RISK"

    if decision_score >= 0.0:
        return "REVIEW"

    return "LOW RISK"


def get_prediction(decision_score, threshold):

    if decision_score >= threshold:
        return "PHISHING"

    if decision_score >= 0.0:
        return "REVIEW REQUIRED"

    return "LIKELY BENIGN"


def get_risk_class(level):

    if level == "HIGH RISK":
        return "danger"

    if level == "REVIEW":
        return "warning"

    return "safe"


# ============================================================
# URL EVIDENCE
# ============================================================

def build_url_evidence(features):

    evidence = []


    # --------------------------------------------------------
    # IP ADDRESS
    # --------------------------------------------------------

    if features["has_ip_address"]:

        evidence.append({
            "title": "IP address detected",
            "description":
                "The URL uses an IP address instead of a conventional domain name.",
            "severity": "high"
        })


    # --------------------------------------------------------
    # @ SYMBOL
    # --------------------------------------------------------

    if features["has_at_symbol"]:

        evidence.append({
            "title": "@ symbol detected",
            "description":
                "The URL contains an @ symbol, which can obscure the actual destination.",
            "severity": "high"
        })


    # --------------------------------------------------------
    # SUSPICIOUS KEYWORDS
    # --------------------------------------------------------

    if features["suspicious_word_count"] > 0:

        evidence.append({
            "title": "Suspicious keywords",
            "description":
                f'{features["suspicious_word_count"]} suspicious keyword(s) detected.',
            "severity": "medium"
        })


    # --------------------------------------------------------
    # SUBDOMAINS
    # --------------------------------------------------------

    if features["num_subdomains"] >= 3:

        evidence.append({
            "title": "Complex subdomain structure",
            "description":
                f'{features["num_subdomains"]} subdomain level(s) detected.',
            "severity": "medium"
        })


    # --------------------------------------------------------
    # URL LENGTH
    # --------------------------------------------------------

    if features["url_length"] > 100:

        evidence.append({
            "title": "Long URL",
            "description":
                f'The URL contains {features["url_length"]} characters.',
            "severity": "medium"
        })


    # --------------------------------------------------------
    # QUERY PARAMETERS
    # --------------------------------------------------------

    if features["num_query_params"] >= 4:

        evidence.append({
            "title": "Many query parameters",
            "description":
                f'{features["num_query_params"]} query parameter(s) detected.',
            "severity": "medium"
        })


    # --------------------------------------------------------
    # HTTPS
    # --------------------------------------------------------

    if not features["has_https"]:

        evidence.append({
            "title": "HTTPS not detected",
            "description":
                "The URL does not use HTTPS.",
            "severity": "medium"
        })


    # --------------------------------------------------------
    # NO OBVIOUS EVIDENCE
    # --------------------------------------------------------

    if not evidence:

        evidence.append({
            "title": "No obvious URL indicators",
            "description":
                "No major suspicious URL patterns were detected.",
            "severity": "low"
        })


    return evidence


# ============================================================
# HTML EVIDENCE
# ============================================================

def build_html_evidence(features):

    evidence = []


    # --------------------------------------------------------
    # PASSWORD INPUT
    # --------------------------------------------------------

    if features["num_password_inputs"] > 0:

        evidence.append({
            "title": "Password field detected",
            "description":
                f'{features["num_password_inputs"]} password input(s) found.',
            "severity": "medium"
        })


    # --------------------------------------------------------
    # EXTERNAL FORM ACTION
    # --------------------------------------------------------

    if features["has_external_form_action"]:

        evidence.append({
            "title": "External form destination",
            "description":
                f'{features["num_external_form_actions"]} form action(s) '
                "point outside the current domain.",
            "severity": "high"
        })


    # --------------------------------------------------------
    # IFRAMES
    # --------------------------------------------------------

    if features["num_iframes"] > 0:

        evidence.append({
            "title": "Iframe detected",
            "description":
                f'{features["num_iframes"]} iframe(s) detected.',
            "severity": "low"
        })


    # --------------------------------------------------------
    # HIDDEN INPUTS
    # --------------------------------------------------------

    if features["num_hidden_inputs"] > 0:

        evidence.append({
            "title": "Hidden inputs",
            "description":
                f'{features["num_hidden_inputs"]} hidden input(s) detected.',
            "severity": "low"
        })


    # --------------------------------------------------------
    # EXTERNAL LINKS
    # --------------------------------------------------------

    if features["num_external_links"] > 0:

        evidence.append({
            "title": "External links",
            "description":
                f'{features["num_external_links"]} external link(s) detected.',
            "severity": "low"
        })


    # --------------------------------------------------------
    # FORMS
    # --------------------------------------------------------

    if features["num_forms"] > 0:

        evidence.append({
            "title": "Forms detected",
            "description":
                f'{features["num_forms"]} form(s) found.',
            "severity": "low"
        })


    # --------------------------------------------------------
    # NO OBVIOUS EVIDENCE
    # --------------------------------------------------------

    if not evidence:

        evidence.append({
            "title": "No major HTML indicators",
            "description":
                "No major suspicious structural indicators were detected.",
            "severity": "low"
        })


    return evidence


# ============================================================
# MODEL 1
# URL TF-IDF
# ============================================================

def predict_url_tfidf(url):

    tfidf_features = url_tfidf.transform(
        [url]
    )

    decision_score = float(
        url_svm.decision_function(
            tfidf_features
        )[0]
    )

    return decision_score


# ============================================================
# MODEL 2
# URL + ENGINEERED FEATURES
# ============================================================

def predict_url_engineered(url):

    url_features = extract_url_features(
        url
    )

    feature_df = pd.DataFrame(
        [url_features],
        columns=URL_FEATURE_COLUMNS
    )

    scaled_features = url_scaler.transform(
        feature_df
    )

    tfidf_features = combined_tfidf.transform(
        [url]
    )

    combined_features = hstack(
        [
            tfidf_features,
            csr_matrix(
                scaled_features
            )
        ],
        format="csr"
    )

    decision_score = float(
        combined_svm.decision_function(
            combined_features
        )[0]
    )

    return (
        decision_score,
        url_features
    )


# ============================================================
# MODEL 3
# URL + HTML
# ============================================================

def predict_url_html(
    url,
    html
):

    url_features = extract_url_features(
        url
    )

    html_features = extract_html_features(
        html,
        url
    )

    url_df = pd.DataFrame(
        [url_features],
        columns=URL_FEATURE_COLUMNS
    )

    html_df = pd.DataFrame(
        [html_features],
        columns=html_feature_columns
    )

    url_scaled = url_scaler.transform(
        url_df
    )

    html_scaled = html_scaler.transform(
        html_df
    )

    tfidf_features = url_html_tfidf.transform(
        [url]
    )

    combined_features = hstack(
        [
            tfidf_features,
            csr_matrix(
                url_scaled
            ),
            csr_matrix(
                html_scaled
            )
        ],
        format="csr"
    )

    decision_score = float(
        url_html_svm.decision_function(
            combined_features
        )[0]
    )

    return (
        decision_score,
        url_features,
        html_features
    )


# ============================================================
# HOME PAGE
# ============================================================

@app.get(
    "/",
    response_class=HTMLResponse
)
async def home(
    request: Request
):

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "request": request,
            "result": None
        }
    )


# ============================================================
# SCAN
# ============================================================

@app.post(
    "/scan",
    response_class=HTMLResponse
)
async def scan(
    request: Request,
    url: str = Form(...),
    model_choice: str = Form(""),
    html: str = Form("")
):

    # --------------------------------------------------------
    # NORMALIZE URL
    # --------------------------------------------------------

    url = normalize_url(url)

    if not model_choice:
        model_choice = "url_html" if html.strip() else "url_tfidf"


    # --------------------------------------------------------
    # EMPTY URL
    # --------------------------------------------------------

    if not url:

        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "request": request,
                "result": {
                    "error":
                        "Please enter a URL."
                }
            }
        )


    # --------------------------------------------------------
    # INITIALIZE
    # --------------------------------------------------------

    decision_score = None

    url_features = None

    html_features = None

    evidence = []

    feature_summary = {}


    # ========================================================
    # MODEL 1
    # URL TF-IDF
    # ========================================================

    if model_choice == "url_tfidf":

        decision_score = predict_url_tfidf(
            url
        )

        model_name = (
            "URL TF-IDF"
        )

        model_description = (
            "Character-level TF-IDF + Linear SVM"
        )

        url_features = extract_url_features(
            url
        )

        evidence = build_url_evidence(
            url_features
        )

        feature_summary = {

            "URL length":
                url_features[
                    "url_length"
                ],

            "Hostname length":
                url_features[
                    "hostname_length"
                ],

            "Path length":
                url_features[
                    "path_length"
                ],

            "Dots":
                url_features[
                    "num_dots"
                ],

            "Digits":
                url_features[
                    "num_digits"
                ],

            "HTTPS":
                (
                    "Yes"
                    if url_features[
                        "has_https"
                    ]
                    else "No"
                )
        }


    # ========================================================
    # MODEL 2
    # URL + ENGINEERED FEATURES
    # ========================================================

    elif model_choice == "url_engineered":

        (
            decision_score,
            url_features
        ) = predict_url_engineered(
            url
        )

        model_name = (
            "URL + Engineered Features"
        )

        model_description = (
            "Character TF-IDF + engineered URL features + Linear SVM"
        )

        evidence = build_url_evidence(
            url_features
        )

        feature_summary = {

            "URL length":
                url_features[
                    "url_length"
                ],

            "Hostname length":
                url_features[
                    "hostname_length"
                ],

            "Path length":
                url_features[
                    "path_length"
                ],

            "Subdomains":
                url_features[
                    "num_subdomains"
                ],

            "Query parameters":
                url_features[
                    "num_query_params"
                ],

            "Suspicious keywords":
                url_features[
                    "suspicious_word_count"
                ],

            "IP address":
                (
                    "Yes"
                    if url_features[
                        "has_ip_address"
                    ]
                    else "No"
                ),

            "HTTPS":
                (
                    "Yes"
                    if url_features[
                        "has_https"
                    ]
                    else "No"
                )
        }


    # ========================================================
    # MODEL 3
    # URL + HTML
    # ========================================================

    elif model_choice == "url_html":

        if not html.strip():

            return templates.TemplateResponse(
                request=request,
                name="index.html",
                context={
                    "request": request,
                    "result": {
                        "error":
                            "HTML is required when using the URL + HTML model."
                    }
                }
            )


        (
            decision_score,
            url_features,
            html_features
        ) = predict_url_html(
            url,
            html
        )

        model_name = (
            "URL + HTML"
        )

        model_description = (
            "URL TF-IDF + engineered URL features + HTML structural features + Linear SVM"
        )

        url_evidence = build_url_evidence(
            url_features
        )

        html_evidence = build_html_evidence(
            html_features
        )

        evidence = (
            url_evidence
            +
            html_evidence
        )

        feature_summary = {

            "URL length":
                url_features[
                    "url_length"
                ],

            "Subdomains":
                url_features[
                    "num_subdomains"
                ],

            "Query parameters":
                url_features[
                    "num_query_params"
                ],

            "Suspicious keywords":
                url_features[
                    "suspicious_word_count"
                ],

            "HTML length":
                html_features[
                    "html_length"
                ],

            "Forms":
                html_features[
                    "num_forms"
                ],

            "Inputs":
                html_features[
                    "num_inputs"
                ],

            "Password inputs":
                html_features[
                    "num_password_inputs"
                ],

            "External links":
                html_features[
                    "num_external_links"
                ],

            "Scripts":
                html_features[
                    "num_scripts"
                ],

            "Iframes":
                html_features[
                    "num_iframes"
                ]
        }


    # ========================================================
    # INVALID MODEL
    # ========================================================

    else:

        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "request": request,
                "result": {
                    "error":
                        "Invalid model selection."
                }
            }
        )


    # ========================================================
    # ACTUAL CLASSIFICATION
    # ========================================================
    #
    # This is the important part.
    #
    # The model produces a decision score.
    #
    # We compare it against the selected model's threshold:
    #
    # score >= threshold -> PHISHING
    # score <  threshold -> LIKELY BENIGN
    #
    # ========================================================

    classification_threshold = CLASSIFICATION_THRESHOLDS[model_choice]

    prediction = get_prediction(
        decision_score,
        classification_threshold,
    )

    reputation_result = await run_in_threadpool(
        lookup_safe_browsing,
        url,
        GOOGLE_SAFE_BROWSING_API_KEY,
    )

    if reputation_result.status == "matched":
        prediction = "PHISHING"
        evidence = [
            item for item in evidence
            if item["title"] != "No obvious URL indicators"
        ]
        evidence.insert(0, {
            "title": "Threat intelligence match",
            "description": (
                "Google Safe Browsing lists this URL as {}.".format(
                    ", ".join(reputation_result.threat_types)
                )
            ),
            "severity": "high",
        })


    # ========================================================
    # VISUAL RISK INDICATOR
    # ========================================================
    #
    # This does NOT control the classification.
    #
    # A positive score below the stricter URL-only cutoff is
    # shown as needing review rather than as a phishing verdict.
    # ========================================================

    risk_level = get_risk_level(
        decision_score,
        classification_threshold,
    )

    if reputation_result.status == "matched":
        risk_level = "HIGH RISK"

    risk_class = get_risk_class(
        risk_level
    )

    reputation_messages = {
        "matched": "Listed by Google Safe Browsing.",
        "not_listed": (
            "No Google Safe Browsing match. This does not confirm the site is safe."
        ),
        "not_configured": (
            "Google Safe Browsing is not configured; only local model analysis ran."
        ),
        "unavailable": (
            "{} Local model analysis is shown.".format(
                reputation_result.detail
            )
        ),
    }


    # ========================================================
    # RESULT
    # ========================================================

    result = {

        "url":
            url,

        "prediction":
            prediction,

        "decision_score":
            round(
                decision_score,
                4
            ),

        "classification_threshold":
            classification_threshold,

        "risk_level":
            risk_level,

        "risk_class":
            risk_class,

        "model_choice":
            model_choice,

        "model_name":
            model_name,

        "model_description":
            model_description,

        "reputation_status":
            reputation_result.status,

        "reputation_message":
            reputation_messages[reputation_result.status],

        "evidence":
            evidence,

        "feature_summary":
            feature_summary
    }


    # ========================================================
    # RENDER RESULT
    # ========================================================

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "request": request,
            "result": result,
            "selected_model": model_choice
        }
    )