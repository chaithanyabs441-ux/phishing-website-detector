import sys
from pathlib import Path

sys.path.append(
    str(
        Path(__file__).resolve().parent
    )
)

from html_features import extract_html_features


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


features = extract_html_features(
    sample_html,
    "https://mywebsite.com/login"
)


print("\nHTML FEATURES")
print("=" * 40)

for name, value in features.items():
    print(f"{name}: {value}")