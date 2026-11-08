"""Migration parity checks for the existing version 1 feature functions."""

import re
import unittest
from urllib.parse import urlparse

from bs4 import BeautifulSoup

from src.features.html_features import extract_html_features
from src.features.url_features import extract_url_features


SUSPICIOUS_WORDS = [
    "login", "signin", "verify", "verification", "secure", "account",
    "update", "confirm", "password", "bank", "wallet", "payment",
    "invoice", "auth",
]


def legacy_benchmark_url_features(url):
    """Reference the duplicate evaluator implementation before its removal."""
    if not isinstance(url, str):
        url = ""

    try:
        parsed = urlparse(url)
        hostname = parsed.hostname or ""
        path = parsed.path or ""
        query = parsed.query or ""
    except Exception:
        hostname = path = query = ""

    url_lower = url.lower()
    suspicious_word_count = sum(word in url_lower for word in SUSPICIOUS_WORDS)
    ip_pattern = r"^(?:\d{1,3}\.){3}\d{1,3}$"
    has_ip = int(re.match(ip_pattern, hostname) is not None)
    num_subdomains = max(0, len(hostname.split(".")) - 2) if hostname else 0

    return {
        "url_length": len(url),
        "hostname_length": len(hostname),
        "path_length": len(path),
        "num_dots": url.count("."),
        "num_hyphens": url.count("-"),
        "num_digits": sum(c.isdigit() for c in url),
        "num_special_chars": sum(not c.isalnum() for c in url),
        "num_subdomains": num_subdomains,
        "num_query_params": query.count("&") + 1 if query else 0,
        "has_at_symbol": int("@" in url),
        "has_ip_address": has_ip,
        "has_https": int(url_lower.startswith("https://")),
        "suspicious_word_count": suspicious_word_count,
    }


def legacy_repair_html_features(html, page_url):
    """Reference the duplicate HTML repair implementation before its removal."""
    if not isinstance(html, str):
        html = ""

    soup = BeautifulSoup(html, "html.parser")
    forms = soup.find_all("form")
    inputs = soup.find_all("input")
    links = soup.find_all("a")
    images = soup.find_all("img")
    scripts = soup.find_all("script")
    iframes = soup.find_all("iframe")
    password_inputs = soup.find_all("input", attrs={"type": "password"})
    email_inputs = soup.find_all("input", attrs={"type": "email"})
    hidden_inputs = soup.find_all("input", attrs={"type": "hidden"})

    def safe_domain(url):
        try:
            return urlparse(url).netloc.lower()
        except Exception:
            return ""

    page_domain = safe_domain(page_url)
    external_links = sum(
        1 for link in links
        if isinstance(link.get("href"), str)
        and link["href"].lower().startswith("http")
        and safe_domain(link["href"])
        and safe_domain(link["href"]) != page_domain
    )
    external_form_actions = sum(
        1 for form in forms
        if isinstance(form.get("action"), str)
        and form["action"].lower().startswith("http")
        and safe_domain(form["action"])
        and safe_domain(form["action"]) != page_domain
    )

    return {
        "html_length": len(html),
        "num_forms": len(forms),
        "num_inputs": len(inputs),
        "num_links": len(links),
        "num_images": len(images),
        "num_scripts": len(scripts),
        "num_iframes": len(iframes),
        "num_password_inputs": len(password_inputs),
        "num_email_inputs": len(email_inputs),
        "num_hidden_inputs": len(hidden_inputs),
        "num_external_links": external_links,
        "num_external_form_actions": external_form_actions,
        "has_password_field": int(bool(password_inputs)),
        "has_form": int(bool(forms)),
        "has_external_form_action": int(bool(external_form_actions)),
    }


class FeatureParityTests(unittest.TestCase):
    def test_url_v1_matches_previous_evaluator_for_standard_urls(self):
        urls = [
            "https://example.com/login?next=home",
            "http://www.example.co.uk/path?x=1&y=2",
            "https://185.150.32.190/path",
            "ftp://ftp.example.org/file",
        ]

        for url in urls:
            with self.subTest(url=url):
                self.assertEqual(
                    extract_url_features(url),
                    legacy_benchmark_url_features(url),
                )

    def test_html_v1_matches_previous_repair_extractor(self):
        examples = [
            (
                "https://example.com/",
                "<form action='https://other.test/post'>"
                "<input type='password'><input type='email'>"
                "</form><a href='https://other.test/'>away</a>",
            ),
            (
                "https://example.com/",
                "<html><body><img src='/logo.png'><script></script>"
                "<iframe src='/frame'></iframe></body></html>",
            ),
            ("not a url", ""),
        ]

        for page_url, html in examples:
            with self.subTest(page_url=page_url, html=html):
                self.assertEqual(
                    extract_html_features(html, page_url),
                    legacy_repair_html_features(html, page_url),
                )


if __name__ == "__main__":
    unittest.main()
