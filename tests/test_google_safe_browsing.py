import io
import json
import unittest
from urllib.error import HTTPError
from unittest.mock import patch

from src.reputation.google_safe_browsing import lookup_url


class GoogleSafeBrowsingTests(unittest.TestCase):
    def test_missing_key_disables_external_lookup_explicitly(self):
        result = lookup_url("https://example.com/", "")

        self.assertEqual(result.status, "not_configured")

    @patch("src.reputation.google_safe_browsing.urlopen")
    def test_threat_match_is_reported(self, mock_urlopen):
        mock_urlopen.return_value = io.BytesIO(json.dumps({
            "matches": [
                {"threatType": "SOCIAL_ENGINEERING"},
            ],
        }).encode("utf-8"))

        result = lookup_url(
            "https://example.com/login",
            "test-key",
        )

        self.assertEqual(result.status, "matched")
        self.assertEqual(result.threat_types, ("SOCIAL_ENGINEERING",))
        request = mock_urlopen.call_args.args[0]
        self.assertIn("https://example.com/login", request.data.decode("utf-8"))

    @patch("src.reputation.google_safe_browsing.urlopen")
    def test_empty_response_is_not_listed_not_confirmed_safe(self, mock_urlopen):
        mock_urlopen.return_value = io.BytesIO(b"{}")

        result = lookup_url("https://example.com/", "test-key")

        self.assertEqual(result.status, "not_listed")

    @patch("src.reputation.google_safe_browsing.urlopen")
    def test_api_error_is_explicitly_unavailable(self, mock_urlopen):
        mock_urlopen.side_effect = HTTPError(
            "https://safebrowsing.googleapis.com/",
            403,
            "Forbidden",
            {},
            None,
        )

        result = lookup_url("https://example.com/", "test-key")

        self.assertEqual(result.status, "unavailable")
        self.assertIn("403", result.detail)


if __name__ == "__main__":
    unittest.main()
