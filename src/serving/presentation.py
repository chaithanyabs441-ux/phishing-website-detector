"""Verdict labels and evidence displayed by the scan interfaces."""

def get_prediction(decision_score, threshold):
    if decision_score >= threshold:
        return "PHISHING"
    if decision_score >= 0:
        return "REVIEW REQUIRED"
    return "LIKELY BENIGN"


def get_risk_level(decision_score, threshold):
    if decision_score >= threshold:
        return "HIGH RISK"
    if decision_score >= 0:
        return "REVIEW"
    return "LOW RISK"


def get_risk_class(level):
    if level in {"BLOCK", "HIGH RISK"}:
        return "danger"
    if level in {"WARN", "REVIEW", "NEEDS PAGE CONTENT"}:
        return "warning"
    if level in {"NO STRONG SIGNAL", "LOW RISK"}:
        return "safe"
    return "safe"


def build_evidence(url_features, html_features, registered_domain, allowlisted):
    evidence = []
    if url_features.get("has_ip_address"):
        evidence.append({"title": "IP address in URL", "description": "The URL uses an IP address as its host.", "severity": "high"})
    if url_features.get("has_at_symbol"):
        evidence.append({"title": "@ symbol in URL", "description": "An @ symbol can obscure the actual destination.", "severity": "high"})
    if url_features.get("suspicious_word_count", 0):
        evidence.append({"title": "Account-related URL terms", "description": "The URL contains terms often used in credential prompts.", "severity": "medium"})
    if url_features.get("is_punycode_or_idn"):
        evidence.append({"title": "Internationalized hostname", "description": "The hostname contains IDN or punycode characters.", "severity": "medium"})
    if html_features:
        if html_features.get("html_missing"):
            evidence.append({"title": "Page HTML not provided", "description": "The analysis could not inspect page structure.", "severity": "low"})
        if html_features.get("num_password_inputs"):
            evidence.append({"title": "Password field", "description": "A password input is present in the supplied page HTML.", "severity": "medium"})
        if html_features.get("num_form_action_domain_mismatches"):
            evidence.append({"title": "External form destination", "description": "A form submits to a different registered domain.", "severity": "high"})
        if html_features.get("num_external_domains"):
            evidence.append({"title": "External page resources", "description": "The page references resources or links on other domains.", "severity": "low"})
        if html_features.get("has_javascript_redirect"):
            evidence.append({"title": "JavaScript redirect pattern", "description": "The supplied script text contains a location redirect pattern.", "severity": "medium"})
    if allowlisted:
        evidence.append({"title": "Local popular-domain list", "description": "The registered domain is on the local context list; this is not proof that this URL or page is safe.", "severity": "low"})
    if not evidence:
        evidence.append({"title": "No prominent structural indicator", "description": "No listed URL or HTML indicator was found.", "severity": "low"})
    return evidence
