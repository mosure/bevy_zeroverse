#!/usr/bin/env python3
"""Compile the repository paper and stage a self-contained project-page download."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import zipfile

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
TEX = ROOT / "tex"
BUILD = ROOT / "out/project_page/latex"
TARGET = ROOT / "www/project/static/papers"
MEDIA = ROOT / "www/project/static/media"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    for folder in (BUILD, TARGET, MEDIA):
        folder.mkdir(parents=True, exist_ok=True)
    subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error",
                    f"-outdir={BUILD}", "bevy_zeroverse.tex"], cwd=TEX, check=True)
    pdf = TARGET / "bevy_zeroverse.pdf"
    shutil.copyfile(BUILD / pdf.name, pdf)
    subprocess.run(["pdftoppm", "-f", "1", "-singlefile", "-scale-to", "1000", "-png",
                    str(pdf), str(BUILD / "whitepaper")], check=True)
    Image.open(BUILD / "whitepaper.png").save(MEDIA / "whitepaper.webp", quality=92, method=6)
    sources = sorted(p for p in TEX.rglob("*") if p.suffix in {".tex", ".sty", ".bib", ".jpg", ".png"})
    with zipfile.ZipFile(TARGET / "whitepaper-source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sources:
            archive.write(path, str(path.relative_to(TEX)))
        archive.writestr("BUILD.txt", "Run latexmk -pdf bevy_zeroverse.tex in this directory.\n"
                         "Appearance/performance tables measure generator v18. Expanded camera/co-visibility results and the gallery use v20; the motion illustration retains v19.\n")
    (TARGET / "provenance.json").write_text(json.dumps({
        "pdf_sha256": sha(pdf),
        "sources": {str(p.relative_to(ROOT)): sha(p) for p in sources},
        "source_archive_sha256": sha(TARGET / "whitepaper-source.zip"),
        "build": "latexmk -pdf -interaction=nonstopmode -halt-on-error bevy_zeroverse.tex",
        "scope": "v20 expanded camera and co-visibility evaluation; v18 appearance/performance baseline; v20 matched gallery and retained v19 motion illustration; technical report"
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
