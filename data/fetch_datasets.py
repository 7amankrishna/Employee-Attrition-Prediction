#!/usr/bin/env python3
"""Re-download the external attrition datasets from their GitHub mirrors.

Run from the repository root:

    python3 data/fetch_datasets.py           # download + verify
    python3 data/fetch_datasets.py --check   # verify only

Only github.com / codeload.github.com are contacted. Each file is verified
against the SHA-256 recorded in data/README.md.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"

# (output file, codeload tarball URL, path inside the tarball, sha256)
SOURCES = [
    (
        "hr-comma-sep.csv",
        "https://codeload.github.com/KunalKurve/Machine-Learning/tar.gz/HEAD",
        "Machine-Learning-HEAD/Cases/human-resources-analytics/HR_comma_sep.csv",
        "2f7028186426cdf4924d4c7b262c3f83bbca85eb14288add46b120d740e6a03b",
    ),
    (
        "av-job-change-train.csv",
        "https://codeload.github.com/aasu14/Data-Science-Hackathon-And-Competition/tar.gz/HEAD",
        "Data-Science-Hackathon-And-Competition-HEAD/AnalyticsVidhya JanataHack HR Analytics/train_jqd04QH.csv",
        "591b300fcc58356f4b142f4b978ad42168f0a76663596d086a9592ccbb3a946c",
    ),
    (
        "academic-faculty.csv",
        "https://codeload.github.com/allisonmorgan/academic_census/tar.gz/HEAD",
        "academic_census-HEAD/data/faculty.csv",
        "99fd9cecd66799566df9967544a85c45dee91779119ad3745bec0a38572e6ede",
    ),
    (
        "academic-department-attrition.csv",
        "https://codeload.github.com/allisonmorgan/academic_census/tar.gz/HEAD",
        "academic_census-HEAD/data/attrition.csv",
        "6a93edbc7ae9f9f737bfbe02734421e66eee549af85d5ca02de46e617a88a435",
    ),
]


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "arena-research-fetch/1.0"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        return resp.read()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="only verify existing files")
    args = ap.parse_args()

    ok = True
    for name, url, inner, digest in SOURCES:
        out = DATA_DIR / name
        if not args.check:
            print(f"[fetch] {name} <- {url}")
            with tempfile.TemporaryDirectory() as td:
                tgz = Path(td) / "repo.tar.gz"
                tgz.write_bytes(fetch(url))
                with tarfile.open(tgz, "r:gz") as tf:
                    member = tf.extractfile(inner)
                    if member is None:
                        print(f"  !! {inner} not found in archive")
                        ok = False
                        continue
                    out.write_bytes(member.read())
            # normalize CRLF -> LF to match the committed copies
            text = out.read_text(encoding="utf-8", errors="replace").replace("\r\n", "\n")
            out.write_text(text, encoding="utf-8")
        if not out.exists():
            print(f"[missing] {name}")
            ok = False
            continue
        actual = sha256_of(out)
        status = "OK " if actual == digest else "MISMATCH"
        if actual != digest:
            ok = False
        print(f"[{status}] {name}  sha256={actual}")
    print("ALL VERIFIED" if ok else "VERIFICATION FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
