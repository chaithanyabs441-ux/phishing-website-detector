from bs4 import BeautifulSoup
from urllib.parse import urlparse


# ============================================================
# SAFE DOMAIN PARSER
# ============================================================

def safe_domain(url):

    try:
        return urlparse(url).netloc.lower()

    except Exception:
        return ""


# ============================================================
# HTML FEATURE EXTRACTION
# ============================================================

def extract_html_features(html, page_url):

    # --------------------------------------------------------
    # SAFETY CHECK
    # --------------------------------------------------------

    if not isinstance(html, str):
        html = ""


    # --------------------------------------------------------
    # PARSE HTML
    # --------------------------------------------------------

    soup = BeautifulSoup(
        html,
        "html.parser"
    )


    # --------------------------------------------------------
    # BASIC HTML ELEMENTS
    # --------------------------------------------------------

    forms = soup.find_all("form")

    inputs = soup.find_all("input")

    links = soup.find_all("a")

    images = soup.find_all("img")

    scripts = soup.find_all("script")

    iframes = soup.find_all("iframe")


    # --------------------------------------------------------
    # INPUT TYPES
    # --------------------------------------------------------

    password_inputs = soup.find_all(
        "input",
        attrs={
            "type": "password"
        }
    )

    email_inputs = soup.find_all(
        "input",
        attrs={
            "type": "email"
        }
    )

    hidden_inputs = soup.find_all(
        "input",
        attrs={
            "type": "hidden"
        }
    )


    # --------------------------------------------------------
    # PAGE DOMAIN
    # --------------------------------------------------------

    page_domain = safe_domain(
        page_url
    )


    # --------------------------------------------------------
    # EXTERNAL LINKS
    # --------------------------------------------------------

    external_links = 0

    for link in links:

        href = link.get("href")

        if not href:
            continue

        if href.lower().startswith("http"):

            link_domain = safe_domain(
                href
            )

            if (
                link_domain
                and link_domain != page_domain
            ):

                external_links += 1


    # --------------------------------------------------------
    # EXTERNAL FORM ACTIONS
    # --------------------------------------------------------

    external_form_actions = 0

    for form in forms:

        action = form.get("action")

        if not action:
            continue

        if action.lower().startswith("http"):

            action_domain = safe_domain(
                action
            )

            if (
                action_domain
                and action_domain != page_domain
            ):

                external_form_actions += 1


    # --------------------------------------------------------
    # FEATURES
    # --------------------------------------------------------

    features = {

        "html_length":
        len(html),

        "num_forms":
        len(forms),

        "num_inputs":
        len(inputs),

        "num_links":
        len(links),

        "num_images":
        len(images),

        "num_scripts":
        len(scripts),

        "num_iframes":
        len(iframes),

        "num_password_inputs":
        len(password_inputs),

        "num_email_inputs":
        len(email_inputs),

        "num_hidden_inputs":
        len(hidden_inputs),

        "num_external_links":
        external_links,

        "num_external_form_actions":
        external_form_actions,

        "has_password_field":
        int(
            len(password_inputs) > 0
        ),

        "has_form":
        int(
            len(forms) > 0
        ),

        "has_external_form_action":
        int(
            external_form_actions > 0
        )
    }


    # --------------------------------------------------------
    # RETURN FEATURES
    # --------------------------------------------------------

    return features