import asyncio
import unittest
from unittest.mock import patch

import webapp.app as webapp
from fastapi import Request
from src.reputation.google_safe_browsing import SafeBrowsingResult


SCREENSHOT_URL = "https://notebook.google.com/?icid=home_maincta"
REPORTED_PHISHING_URL = "https://orca-app-2ulpy.ondigitalocean.app"


class ModelThresholdTests(unittest.TestCase):
    def setUp(self):
        self.previous_api_key = webapp.GOOGLE_SAFE_BROWSING_API_KEY
        webapp.GOOGLE_SAFE_BROWSING_API_KEY = ""

    def tearDown(self):
        webapp.GOOGLE_SAFE_BROWSING_API_KEY = self.previous_api_key

    def test_screenshot_url_is_not_labeled_phishing(self):
        score = webapp.predict_url_tfidf(SCREENSHOT_URL)
        prediction = webapp.get_prediction(
            score,
            webapp.CLASSIFICATION_THRESHOLDS["url_tfidf"],
        )

        self.assertAlmostEqual(score, 1.4541, places=3)
        self.assertEqual(prediction, "REVIEW REQUIRED")
        self.assertEqual(webapp.get_risk_level(score, 1.5), "REVIEW")

    def test_reported_phishing_url_is_review_not_benign(self):
        score = webapp.predict_url_tfidf(REPORTED_PHISHING_URL)
        threshold = webapp.CLASSIFICATION_THRESHOLDS["url_tfidf"]

        self.assertAlmostEqual(score, 0.8005, places=3)
        self.assertEqual(
            webapp.get_prediction(score, threshold),
            "REVIEW REQUIRED",
        )
        self.assertEqual(webapp.get_risk_level(score, threshold), "REVIEW")

        response = asyncio.run(
            webapp.scan(
                Request({
                    "type": "http",
                    "method": "POST",
                    "path": "/scan",
                }),
                url=REPORTED_PHISHING_URL,
                model_choice="",
                html="",
            )
        )
        rendered = response.body.decode()
        self.assertIn("REVIEW REQUIRED", rendered)
        self.assertNotIn("LIKELY BENIGN", rendered)

    def test_prediction_distinguishes_phishing_review_and_negative_scores(self):
        threshold = webapp.CLASSIFICATION_THRESHOLDS["url_tfidf"]
        self.assertEqual(webapp.get_prediction(threshold, threshold), "PHISHING")
        self.assertEqual(webapp.get_prediction(0.0001, threshold), "REVIEW REQUIRED")
        self.assertEqual(webapp.get_prediction(0.0, threshold), "REVIEW REQUIRED")
        self.assertEqual(webapp.get_prediction(-0.0001, threshold), "LIKELY BENIGN")

    def test_url_only_threshold_boundary_and_review_band(self):
        threshold = webapp.CLASSIFICATION_THRESHOLDS["url_tfidf"]
        self.assertGreater(threshold, 1.4541)
        self.assertEqual(webapp.get_risk_level(0.0, threshold), "REVIEW")
        self.assertEqual(
            webapp.get_risk_level(threshold - 0.0001, threshold),
            "REVIEW",
        )
        self.assertEqual(
            webapp.get_risk_level(threshold, threshold),
            "HIGH RISK",
        )
        self.assertEqual(
            webapp.get_risk_level(-0.0001, threshold),
            "LOW RISK",
        )

    def test_pasted_html_uses_url_html_model_when_no_model_is_selected(self):
        request = Request({
            "type": "http",
            "method": "POST",
            "path": "/scan",
        })
        response = asyncio.run(
            webapp.scan(
                request,
                url="example.com/login",
                html=(
                    "<html><body><form action='https://other.example/'>"
                    "<input type='password'></form></body></html>"
                ),
            )
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn("URL + HTML", response.body.decode())

    @patch(
        "webapp.app.lookup_safe_browsing",
        return_value=SafeBrowsingResult(
            status="matched",
            threat_types=("SOCIAL_ENGINEERING",),
        ),
    )
    def test_safe_browsing_match_overrides_model_review(self, _mock_lookup):
        response = asyncio.run(
            webapp.scan(
                Request({
                    "type": "http",
                    "method": "POST",
                    "path": "/scan",
                }),
                url=REPORTED_PHISHING_URL,
                model_choice="",
                html="",
            )
        )
        rendered = response.body.decode()

        self.assertIn("<h2>\n\n                    PHISHING", rendered)
        self.assertIn("Threat intelligence match", rendered)
        self.assertIn("Listed by Google Safe Browsing.", rendered)

    def test_unrelated_model_thresholds_remain_unchanged(self):
        self.assertEqual(
            webapp.CLASSIFICATION_THRESHOLDS["url_engineered"],
            0.0,
        )
        self.assertEqual(
            webapp.CLASSIFICATION_THRESHOLDS["url_html"],
            0.0,
        )

    def test_review_template_shows_no_probability_and_threshold(self):
        template = webapp.templates.get_template("index.html")
        response = template.render(
            request=Request({"type": "http", "method": "GET", "path": "/"}),
            result={
                "prediction": "REVIEW REQUIRED",
                "url": SCREENSHOT_URL,
                "risk_level": "REVIEW",
                "risk_class": "warning",
                "model_name": "URL TF-IDF",
                "decision_score": 1.4541,
                "classification_threshold": 1.5,
                "evidence": [],
                "feature_summary": {},
            },
            selected_model="url_tfidf",
        )
        self.assertIn("Borderline signal", response)
        self.assertIn("1.5", response)
        self.assertIn("URL TF-IDF", response)
        self.assertNotIn("81.1", response)


if __name__ == "__main__":
    unittest.main()
