import unittest

from src.features.url_features import (
    URL_FEATURE_COLUMNS_V1,
    URL_FEATURE_COLUMNS_V2,
    extract_url_features,
    extract_url_features_v1,
    extract_url_features_v2,
)


class UrlFeatureV2Tests(unittest.TestCase):
    def test_v1_schema_and_saved_model_behavior_remain_unchanged(self):
        url = "https://user:pw@foo.bar.co.uk:8443/login"
        features = extract_url_features_v1(url)

        self.assertEqual(features, extract_url_features(url))
        self.assertEqual(list(features), URL_FEATURE_COLUMNS_V1)
        self.assertEqual(features["hostname_length"], len("user:pw@foo.bar.co.uk:8443"))
        self.assertEqual(features["num_subdomains"], 2)

    def test_extracts_registered_domain_and_multilabel_suffixes_offline(self):
        uk = extract_url_features_v2(
            "https://user:pw@foo.bar.co.uk:8443/login"
        )
        brazil = extract_url_features_v2("http://shop.example.com.br/path")

        self.assertEqual(uk["hostname_length"], len("foo.bar.co.uk"))
        self.assertEqual(uk["registered_domain"], "bar.co.uk")
        self.assertEqual(uk["public_suffix"], "co.uk")
        self.assertEqual(uk["num_subdomains"], 1)
        self.assertEqual(uk["tld_category"], "country_code_multi")
        self.assertEqual(brazil["registered_domain"], "example.com.br")
        self.assertEqual(brazil["public_suffix"], "com.br")

    def test_token_keywords_shorteners_and_encoded_bytes(self):
        embedded_keyword = extract_url_features_v2(
            "https://myloginaccount.example.com/path"
        )
        separated_keyword = extract_url_features_v2(
            "https://example.com/login/account"
        )
        shortener = extract_url_features_v2("https://bit.ly/a%2F%20")

        self.assertEqual(embedded_keyword["suspicious_word_count"], 0)
        self.assertEqual(separated_keyword["suspicious_word_count"], 2)
        self.assertEqual(shortener["is_shortener"], 1)
        self.assertEqual(shortener["hex_encoding_count"], 2)

    def test_ip_punycode_and_idn_flags(self):
        ipv4 = extract_url_features_v2("http://192.0.2.1:8080/a")
        ipv6 = extract_url_features_v2("https://[2001:db8::1]:8443/a")
        punycode = extract_url_features_v2(
            "https://xn--bcher-kva.example/login"
        )
        unicode_host = extract_url_features_v2("https://bücher.example/login")

        self.assertEqual(ipv4["has_ip_address"], 1)
        self.assertEqual(ipv6["has_ip_address"], 1)
        self.assertEqual(ipv6["registered_domain"], "2001:db8::1")
        self.assertEqual(punycode["is_punycode_or_idn"], 1)
        self.assertEqual(unicode_host["is_punycode_or_idn"], 1)

    def test_empty_malformed_and_very_long_urls_are_deterministic(self):
        for url in ("", "http://[broken", "x" * 10000):
            with self.subTest(length=len(url)):
                first = extract_url_features_v2(url)
                second = extract_url_features_v2(url)
                self.assertEqual(first, second)
                self.assertEqual(list(first), URL_FEATURE_COLUMNS_V2)
                self.assertTrue(all(value is not None for value in first.values()))

        empty = extract_url_features_v2("")
        self.assertEqual(empty["shannon_entropy"], 0.0)
        self.assertEqual(empty["digit_ratio"], 0.0)
        self.assertEqual(empty["letter_ratio"], 0.0)


if __name__ == "__main__":
    unittest.main()
