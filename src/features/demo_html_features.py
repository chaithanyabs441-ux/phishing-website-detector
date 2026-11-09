import sys
from pathlib import Path

sample_html = """
<html>
<head>
    <title>Login</title>
</head>

<body>

    <form action="/login">

        <input type="email">

        <input type="password">

        <input type="hidden">

        <button>Login</button>

    </form>

    <a href="https://example.com">External link</a>

    <script src="app.js"></script>

    <iframe src="frame.html"></iframe>

</body>
</html>
"""


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.features.html_features import extract_html_features

    features = extract_html_features(sample_html, "https://mywebsite.com/login")
    print("\nHTML FEATURES")
    print("=" * 40)
    for name, value in features.items():
        print(f"{name}: {value}")


if __name__ == "__main__":
    main()
