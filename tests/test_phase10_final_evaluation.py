from src.evaluation.evaluate_phase10_final import _feature_frames


def test_phase10_evaluation_builds_both_serving_feature_frames_without_data():
    rows = [{
        "url": "https://example.com/login?next=%2Faccount",
        "html": '<form action="https://example.net"><input type="password"></form>',
    }]

    import pandas as pd

    url_only, url_html, records = _feature_frames(pd.DataFrame(rows))

    assert len(url_only) == 1
    assert len(url_html) == 1
    assert url_only.loc[0, "registered_domain"] == "example.com"
    assert url_html.loc[0, "num_password_inputs"] == 1
    assert records[0]["registered_domain"] == "example.com"
