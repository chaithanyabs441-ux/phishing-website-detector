"""Download only PhreshPhish train/test Parquet shards to data/raw."""

from pathlib import Path

from huggingface_hub import snapshot_download


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw"


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id="phreshphish/phreshphish",
        repo_type="dataset",
        allow_patterns=["data/train-*.parquet", "data/test-*.parquet"],
        local_dir=DATA_DIR,
    )
    shards = sorted((DATA_DIR / "data").glob("*.parquet"))
    print("Downloaded parquet shards:", len(shards))
    print("Destination:", DATA_DIR / "data")


if __name__ == "__main__":
    main()
