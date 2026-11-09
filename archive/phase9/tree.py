# tree.py — Write project structure to project_structure.txt
import os
from pathlib import Path

EXCLUDE_DIRS = {
    '.git', '.venv', 'venv', 'env', '__pycache__', '.vscode',
    '.idea', 'node_modules', '.pytest_cache', '.mypy_cache',
    'dist', 'build', '.ipynb_checkpoints',
}
EXCLUDE_EXTS = {'.pyc', '.pyo', '.pyd', '.log', '.tmp'}
EXCLUDE_FILE_SUFFIXES = ('.egg-info',)


def should_skip(path: Path) -> bool:
    if path.is_dir():
        return path.name in EXCLUDE_DIRS
    if path.suffix in EXCLUDE_EXTS:
        return True
    if path.name.endswith(EXCLUDE_FILE_SUFFIXES):
        return True
    return False


def build_tree(root: Path, prefix: str = "") -> list[str]:
    lines = []
    try:
        entries = sorted(
            [p for p in root.iterdir() if not should_skip(p)],
            key=lambda p: (p.is_file(), p.name.lower()),
        )
    except PermissionError:
        return lines

    for i, entry in enumerate(entries):
        is_last = i == len(entries) - 1
        branch = "└── " if is_last else "├── "
        if entry.is_dir():
            lines.append(f"{prefix}{branch}{entry.name}/")
            lines.extend(build_tree(entry, prefix + ("    " if is_last else "│   ")))
        else:
            lines.append(f"{prefix}{branch}{entry.name}")
    return lines


def main():
    root = Path.cwd()
    out_file = root / "project_structure.txt"

    lines = [f"{root.name}/"]
    lines.extend(build_tree(root))

    out_file.write_text("\n".join(lines), encoding="utf-8")
    print(f"✅ Saved to {out_file}")
    print(f"   {len(lines)} lines written.")


if __name__ == "__main__":
    main()