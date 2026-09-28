#!/usr/bin/env python3
"""Compile the repository paper and stage a self-contained project-page download."""
import hashlib
import json
import re
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
    # Package only the paper's current dependency closure, keeping unused
    # historical figures and tables out of the downloadable whitepaper source.
    pending = [TEX / "bevy_zeroverse.tex", TEX / "arxiv.sty", TEX / "references.bib"]
    used = set()
    while pending:
        source = pending.pop()
        if source in used:
            continue
        assert source.is_file(), source
        used.add(source)
        if source.suffix == ".tex":
            for name in re.findall(r"\\(?:input|includegraphics)(?:\[[^\]]*\])?\{([^}]+)\}", source.read_text()):
                dependency = TEX / name
                if not dependency.suffix:
                    dependency = dependency.with_suffix(".tex")
                pending.append(dependency)
    sources = sorted(used)
    with zipfile.ZipFile(TARGET / "whitepaper-source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sources:
            archive.write(path, str(path.relative_to(TEX)))
        archive.writestr("BUILD.txt", "Run latexmk -pdf bevy_zeroverse.tex in this directory.\n"
                         "All population results, baseline sweeps and gallery illustrations use generator v21.\n")
    (TARGET / "provenance.json").write_text(json.dumps({
        "pdf_sha256": sha(pdf),
        "sources": {str(p.relative_to(ROOT)): sha(p) for p in sources},
        "source_archive_sha256": sha(TARGET / "whitepaper-source.zip"),
        "build": "latexmk -pdf -interaction=nonstopmode -halt-on-error bevy_zeroverse.tex",
        "scope": "Generator 21 absolute population and camera-baseline measurements; matched gallery and motion illustration; technical report"
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
