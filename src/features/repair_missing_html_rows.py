from pathlib import Path
import pandas as pd

from bs4 import BeautifulSoup
from urllib.parse import urlparse


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

parquet_folder = Path("data/raw/data")
html_folder = Path("data/processed/html_features")

problem_files = [
    "train-050.parquet",
    "train-054.parquet"
]


# --------------------------------------------------
# SAFE URL PARSER
# --------------------------------------------------

def safe_domain(url):
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""


# --------------------------------------------------
# FEATURE EXTRACTION
# --------------------------------------------------

def extract_html_features(html, page_url):

    if not isinstance(html, str):
        html = ""

    soup = BeautifulSoup(html, "html.parser")

    forms = soup.find_all("form")
    inputs = soup.find_all("input")
    links = soup.find_all("a")
    images = soup.find_all("img")
    scripts = soup.find_all("script")
    iframes = soup.find_all("iframe")

    password_inputs = soup.find_all(
        "input",
        attrs={"type": "password"}
    )

    email_inputs = soup.find_all(
        "input",
        attrs={"type": "email"}
    )

    hidden_inputs = soup.find_all(
        "input",
        attrs={"type": "hidden"}
    )

    page_domain = safe_domain(page_url)

    # --------------------------------------------------
    # EXTERNAL LINKS
    # --------------------------------------------------

    external_links = 0

    for link in links:

        href = link.get("href")

        if not href:
            continue

        if not isinstance(href, str):
            continue

        if href.lower().startswith("http"):

            link_domain = safe_domain(href)

            if link_domain and link_domain != page_domain:
                external_links += 1

    # --------------------------------------------------
    # EXTERNAL FORM ACTIONS
    # --------------------------------------------------

    external_form_actions = 0

    for form in forms:

        action = form.get("action")

        if not action:
            continue

        if not isinstance(action, str):
            continue

        if action.lower().startswith("http"):

            action_domain = safe_domain(action)

            if action_domain and action_domain != page_domain:
                external_form_actions += 1

    # --------------------------------------------------
    # FEATURES
    # --------------------------------------------------

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
        "has_password_field": int(len(password_inputs) > 0),
        "has_form": int(len(forms) > 0),
        "has_external_form_action": int(
            external_form_actions > 0
        )
    }


# --------------------------------------------------
# REPAIR
# --------------------------------------------------

print("=" * 70)
print("REPAIRING MISSING HTML ROWS")
print("=" * 70)


for filename in problem_files:

    parquet_file = parquet_folder / filename

    csv_file = html_folder / (
        parquet_file.stem + "_html_features.csv"
    )

    print()
    print("-" * 70)
    print("PROCESSING:", filename)
    print("-" * 70)

    original = pd.read_parquet(
        parquet_file,
        columns=["url", "label", "html"]
    )

    processed = pd.read_csv(csv_file)

    original_urls = set(
        original["url"].astype(str)
    )

    processed_urls = set(
        processed["url"].astype(str)
    )

    missing_urls = original_urls - processed_urls

    print("Missing rows:", len(missing_urls))

    for url in missing_urls:

        row = original[
            original["url"].astype(str) == url
        ].iloc[0]

        print()
        print("Repairing:")
        print("URL   :", row["url"])
        print("LABEL :", row["label"])

        features = extract_html_features(
            row["html"],
            row["url"]
        )

        features["url"] = row["url"]
        features["label"] = row["label"]

        new_row = pd.DataFrame([features])

        new_row = new_row[processed.columns]

        processed = pd.concat(
            [processed, new_row],
            ignore_index=True
        )

    processed.to_csv(
        csv_file,
        index=False
    )

    print()
    print("SAVED:", csv_file)
    print("New row count:", len(processed))


print()
print("=" * 70)
print("REPAIR COMPLETE")
print("=" * 70)