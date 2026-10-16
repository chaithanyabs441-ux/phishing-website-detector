from huggingface_hub import snapshot_download
from pathlib import Path


# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Where benchmark files will be stored
BENCHMARK_DIR = PROJECT_ROOT / "data" / "benchmarks"


print("=" * 60)
print("PHRESHPHISH BENCHMARK DOWNLOAD")
print("=" * 60)

print()
print("Project root:")
print(PROJECT_ROOT)

print()
print("Benchmark directory:")
print(BENCHMARK_DIR)

BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)

print()
print("Downloading official PhreshPhish benchmark files...")
print("This downloads ONLY the benchmark folder.")
print()

snapshot_download(
    repo_id="phreshphish/phreshphish",
    repo_type="dataset",
    allow_patterns=[
        "benchmark/**"
    ],
    local_dir=BENCHMARK_DIR
)

print()
print("=" * 60)
print("DOWNLOAD COMPLETE")
print("=" * 60)

# Count files
files = list(BENCHMARK_DIR.rglob("*"))

txt_files = [
    f for f in files
    if f.is_file() and f.suffix.lower() == ".txt"
]

print()
print("Total files:", len(files))
print("TXT files:", len(txt_files))

print()
print("Benchmark folders:")

for folder in sorted(BENCHMARK_DIR.rglob("*")):
    if folder.is_dir():
        print("  ", folder.relative_to(BENCHMARK_DIR))

print()
print("TXT files by folder:")

folders = {}

for file in txt_files:
    relative = file.relative_to(BENCHMARK_DIR)

    if len(relative.parts) > 1:
        folder = relative.parts[0]
    else:
        folder = "ROOT"

    folders.setdefault(folder, 0)
    folders[folder] += 1

for folder, count in sorted(folders.items()):
    print(f"  {folder}: {count}")

print()
print("Done.")