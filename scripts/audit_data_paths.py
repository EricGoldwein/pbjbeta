#!/usr/bin/env python3
"""
Print resolved PBJapp data roots: source, existence, file counts, approximate size.

Read-only diagnostic — does not move, delete, or rewrite pipeline inputs/outputs.

Usage:
  python scripts/audit_data_paths.py
  python scripts/audit_data_paths.py --json
  python scripts/audit_data_paths.py --quick
  python scripts/audit_data_paths.py --list-refs
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data_path_resolver import (  # noqa: E402
    DATA_ROOT_KEYS,
    ENV_KEYS,
    GLOBAL_DATA_ROOT_ENV,
    GLOBAL_REPO_ROOT_ENV,
    resolve_all_data_paths,
    repo_root,
)

SKIP_WALK_DIRS = {
    ".git",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
}

REF_PATTERN = re.compile(
    r"NonNursecsv|standardized_NonNurse|PBJcsv|standardized_PBJ|"
    r"provider_info_extracted|provider_info_normalized|metrics_backups|"
    r"indexed/|deployments/|provider_info/|\bEIN/"
)

REF_EXTS = {
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".json",
    ".yml",
    ".yaml",
    ".md",
    ".sh",
    ".bat",
    ".ps1",
    ".toml",
    ".cfg",
    ".ini",
    ".html",
}


def _human_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    for unit in ("KiB", "MiB", "GiB", "TiB"):
        n /= 1024
        if n < 1024:
            return f"{n:.2f} {unit}"
    return f"{n:.2f} PiB"


def _walk_stats(path: Path, *, quick: bool) -> dict[str, int | bool]:
    if not path.exists():
        return {"exists": False, "files": 0, "dirs": 0, "bytes": 0}

    files = 0
    dirs = 0
    total_bytes = 0
    max_files = 50_000 if quick else 500_000

    try:
        for dirpath, dirnames, filenames in os.walk(path, topdown=True):
            dirnames[:] = [d for d in dirnames if d not in SKIP_WALK_DIRS]
            dirs += len(dirnames)
            for fn in filenames:
                files += 1
                fp = Path(dirpath) / fn
                try:
                    total_bytes += fp.stat().st_size
                except OSError:
                    pass
                if files >= max_files:
                    return {
                        "exists": True,
                        "files": files,
                        "dirs": dirs,
                        "bytes": total_bytes,
                        "truncated": True,
                    }
    except OSError:
        return {"exists": True, "files": files, "dirs": dirs, "bytes": total_bytes, "error": True}

    return {"exists": True, "files": files, "dirs": dirs, "bytes": total_bytes, "truncated": False}


def _collect_code_refs(root: Path) -> dict[str, list[dict[str, object]]]:
    skip_top = set(DATA_ROOT_KEYS.values()) | {
        "Archive",
        ".git",
        "node_modules",
        "__pycache__",
    }
    by_file: dict[str, list[dict[str, object]]] = {}

    for dirpath, dirnames, filenames in os.walk(root):
        top = Path(dirpath).name
        if dirpath != str(root) and top in skip_top:
            dirnames.clear()
            continue
        dirnames[:] = [d for d in dirnames if d not in SKIP_WALK_DIRS and d not in skip_top]
        for fn in filenames:
            p = Path(dirpath) / fn
            if p.suffix.lower() not in REF_EXTS:
                continue
            try:
                text = p.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            rel = str(p.relative_to(root)).replace("\\", "/")
            for i, line in enumerate(text.splitlines(), 1):
                if REF_PATTERN.search(line):
                    by_file.setdefault(rel, []).append({"line": i, "text": line.strip()[:200]})
    return by_file


def _print_table(rows: list[dict[str, object]]) -> None:
    headers = ("key", "resolved", "source", "exists", "files", "size", "notes")
    widths = {h: len(h) for h in headers}
    str_rows: list[dict[str, str]] = []
    for row in rows:
        s = {
            "key": str(row["key"]),
            "resolved": str(row["resolved"]),
            "source": str(row["source"]),
            "exists": "yes" if row["exists"] else "no",
            "files": str(row["files"]),
            "size": str(row["size"]),
            "notes": str(row.get("notes", "")),
        }
        str_rows.append(s)
        for h in headers:
            widths[h] = max(widths[h], len(s[h]))

    def fmt(h: str, val: str) -> str:
        if h == "resolved":
            return val.ljust(widths[h])
        return val.rjust(widths[h]) if h in ("files", "size") else val.ljust(widths[h])

    print(" ".join(h.ljust(widths[h]) for h in headers))
    print(" ".join("-" * widths[h] for h in headers))
    for s in str_rows:
        print(" ".join(fmt(h, s[h]) for h in headers))


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit resolved PBJapp data paths (read-only).")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Cap directory walks (faster on very large trees).",
    )
    parser.add_argument(
        "--list-refs",
        action="store_true",
        help="Also scan repo code (excluding large data dirs) for hardcoded path strings.",
    )
    args = parser.parse_args()

    root = repo_root()
    resolved = resolve_all_data_paths(root=root)
    rows: list[dict[str, object]] = []

    for key, item in resolved.items():
        stats = _walk_stats(item.path, quick=args.quick)
        notes: list[str] = []
        if stats.get("truncated"):
            notes.append("walk truncated")
        if stats.get("error"):
            notes.append("partial walk")
        if item.source == "repo_default" and not stats.get("exists"):
            notes.append(f"override via {ENV_KEYS[key]} or data_paths.local.json")

        rows.append(
            {
                "key": key,
                "resolved": str(item.path),
                "source": item.source,
                "env_var": item.env_var,
                "exists": bool(stats.get("exists")),
                "files": int(stats.get("files", 0)),
                "dirs": int(stats.get("dirs", 0)),
                "bytes": int(stats.get("bytes", 0)),
                "size": _human_bytes(int(stats.get("bytes", 0))),
                "notes": "; ".join(notes),
            }
        )

    payload: dict[str, object] = {
        "repo_root": str(root),
        "global_env": {
            GLOBAL_REPO_ROOT_ENV: os.environ.get(GLOBAL_REPO_ROOT_ENV, ""),
            GLOBAL_DATA_ROOT_ENV: os.environ.get(GLOBAL_DATA_ROOT_ENV, ""),
        },
        "data_roots": rows,
    }

    if args.list_refs:
        payload["code_references"] = _collect_code_refs(root)

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"repo_root: {root}")
        print(
            f"env: {GLOBAL_REPO_ROOT_ENV}={os.environ.get(GLOBAL_REPO_ROOT_ENV, '') or '(unset)'}  "
            f"{GLOBAL_DATA_ROOT_ENV}={os.environ.get(GLOBAL_DATA_ROOT_ENV, '') or '(unset)'}"
        )
        print()
        _print_table(rows)
        if args.list_refs:
            refs = payload["code_references"]
            assert isinstance(refs, dict)
            print()
            print(f"Hardcoded path references in code/docs: {len(refs)} file(s)")
            for rel in sorted(refs)[:40]:
                print(f"  {rel}")
            if len(refs) > 40:
                print(f"  ... and {len(refs) - 40} more (use --json for full list)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
