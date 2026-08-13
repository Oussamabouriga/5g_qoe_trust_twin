"""Build and audit the final IEEE PDF report.

The build is intentionally fail-closed: report assets are regenerated from the
authoritative frozen metrics, the LaTeX source is compiled, and the resulting
PDF is rejected if it is empty, exceeds the assignment's 15-page body limit,
or still contains text from the superseded report.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path

from generate_report_assets import main as generate_report_assets
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "report/ieee_report.tex"
BUILD_DIRECTORY = ROOT / "tmp/report_build"
OUTPUT_DIRECTORY = ROOT / "output/pdf"
OUTPUT = OUTPUT_DIRECTORY / "ieee_report_Bouriga_BenAissa_final.pdf"


def resolve_tectonic(explicit_path: str | None) -> Path:
    """Resolve a Tectonic executable supplied by CLI, PATH, or local tools."""
    candidates = []
    if explicit_path:
        candidates.append(Path(explicit_path))
    if executable := shutil.which("tectonic"):
        candidates.append(Path(executable))
    candidates.append(ROOT / "tmp/tools/tectonic")

    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate.resolve()
    raise FileNotFoundError(
        "Tectonic is required to build the report. Install it or pass "
        "--tectonic /path/to/tectonic."
    )


def run_tectonic(tectonic: Path) -> None:
    """Compile the IEEE source with bibliography support."""
    BUILD_DIRECTORY.mkdir(parents=True, exist_ok=True)
    command = [
        str(tectonic),
        "-X",
        "compile",
        str(SOURCE),
        "--outdir",
        str(BUILD_DIRECTORY),
        "--keep-logs",
        "--keep-intermediates",
    ]
    subprocess.run(command, cwd=ROOT, check=True)


def audit_pdf(path: Path) -> tuple[int, int]:
    """Check page count, text extraction, body length, and stale statements."""
    reader = PdfReader(path)
    page_count = len(reader.pages)
    if page_count == 0:
        raise ValueError("Compiled report contains no pages.")

    page_text = [page.extract_text() or "" for page in reader.pages]
    combined = "\n".join(page_text)
    if len(combined.strip()) < 10_000:
        raise ValueError("Compiled report contains unexpectedly little text.")

    references_page = None
    for index, text in enumerate(page_text, start=1):
        if text.lstrip().lower().startswith("references\n"):
            references_page = index
            break
    if references_page is None:
        raise ValueError("Compiled report does not contain a References section.")
    body_pages = references_page - 1
    if body_pages > 15:
        raise ValueError(f"Report body has {body_pages} pages; the maximum is 15.")

    required_phrases = (
        "State of the Art",
        "Work Description and Research Design",
        "Methodology",
        "Actions Performed and Software Implementation",
        "Results",
        "Suggestions for Improvement",
        "Requirement Compliance Audit",
        "0.8807",
        "0.9549",
        "0.0715",
        "0.0116",
        "8,365",
        "9.60%",
        "30 (100%)",
    )
    missing = [phrase for phrase in required_phrases if phrase not in combined]
    if missing:
        raise ValueError(
            "Compiled report is missing required content: " + ", ".join(missing)
        )

    forbidden_phrases = (
        "0.8970 accuracy",
        "0.8779 F1",
        "15,522",
        "96.67%",
        "must be added before final submission",
        "exact numerical MOS threshold must be verified",
        "GPT-4o mini",
    )
    stale = [phrase for phrase in forbidden_phrases if phrase in combined]
    if stale:
        raise ValueError(
            "Compiled report contains superseded content: " + ", ".join(stale)
        )

    return page_count, body_pages


def main() -> None:
    """Generate report inputs, compile, audit, and publish the final PDF."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tectonic",
        help="Path to a Tectonic executable (otherwise PATH/local tool is used).",
    )
    arguments = parser.parse_args()

    generate_report_assets()
    tectonic = resolve_tectonic(arguments.tectonic)
    run_tectonic(tectonic)

    compiled = BUILD_DIRECTORY / SOURCE.with_suffix(".pdf").name
    if not compiled.is_file():
        raise FileNotFoundError(
            f"Tectonic did not produce the expected PDF: {compiled}"
        )
    page_count, body_pages = audit_pdf(compiled)

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    shutil.copy2(compiled, OUTPUT)
    # Re-open the delivered bytes rather than trusting only the build copy.
    audit_pdf(OUTPUT)

    print(f"Built: {OUTPUT.relative_to(ROOT)}")
    print(f"Pages: {page_count} total; {body_pages} before References")


if __name__ == "__main__":
    main()
