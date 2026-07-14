"""Perform structural checks on the Overleaf-ready report source."""

from __future__ import annotations

import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT = PROJECT_ROOT / "report"
TEX = REPORT / "acl_latex.tex"
BIB = REPORT / "custom.bib"


def _without_comments(text: str) -> str:
    lines: list[str] = []
    for line in text.splitlines():
        match = re.search(r"(?<!\\)%", line)
        lines.append(line[: match.start()] if match else line)
    return "\n".join(lines)


def _validate_braces(text: str) -> None:
    depth = 0
    escaped = False
    for character in _without_comments(text):
        if escaped:
            escaped = False
            continue
        if character == "\\":
            escaped = True
        elif character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth < 0:
                raise AssertionError("LaTeX source contains an unmatched closing brace")
    if depth:
        raise AssertionError(f"LaTeX source contains {depth} unmatched opening brace(s)")


def _validate_environments(text: str) -> None:
    stack: list[str] = []
    for action, environment in re.findall(r"\\(begin|end)\{([^}]+)\}", text):
        if action == "begin":
            stack.append(environment)
        elif not stack or stack.pop() != environment:
            raise AssertionError(f"mismatched LaTeX environment near {environment}")
    if stack:
        raise AssertionError(f"unclosed LaTeX environments: {stack}")


def main() -> None:
    tex = TEX.read_text(encoding="utf-8")
    bib = BIB.read_text(encoding="utf-8")
    for forbidden in ("TODO", "todotext", "GITHUB-URL", "RESULT TABLE"):
        if forbidden in tex or forbidden in bib:
            raise AssertionError(f"draft placeholder remains: {forbidden}")

    _validate_braces(tex)
    _validate_environments(tex)

    citation_keys = {
        key.strip()
        for group in re.findall(r"\\cite[tp]?\{([^}]+)\}", tex)
        for key in group.split(",")
    }
    bibliography_keys = set(re.findall(r"@\w+\{([^,]+),", bib))
    missing_citations = citation_keys - bibliography_keys
    if missing_citations:
        raise AssertionError(f"missing bibliography keys: {sorted(missing_citations)}")

    labels = set(re.findall(r"\\label\{([^}]+)\}", tex))
    references = set(re.findall(r"\\ref\{([^}]+)\}", tex))
    missing_labels = references - labels
    if missing_labels:
        raise AssertionError(f"missing labels: {sorted(missing_labels)}")
    if len(labels) != len(re.findall(r"\\label\{([^}]+)\}", tex)):
        raise AssertionError("duplicate LaTeX labels detected")

    graphics = re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", tex)
    if not graphics:
        raise AssertionError("report contains no figures")
    missing_graphics = [path for path in graphics if not (REPORT / path).is_file()]
    if missing_graphics:
        raise AssertionError(f"missing figure files: {missing_graphics}")
    non_pdf_graphics = [path for path in graphics if Path(path).suffix.lower() != ".pdf"]
    if non_pdf_graphics:
        raise AssertionError(f"report figures must be PDF: {non_pdf_graphics}")

    required_sections = (
        "Introduction",
        "Background and Related Work",
        "Experimental Setup",
        "Results",
        "Discussion",
        "Limitations",
        "Conclusion",
        "Prompt Templates",
        "Full Experimental Configuration",
        "Additional Results",
    )
    positions = [tex.index(f"{{{section}}}") for section in required_sections]
    if positions != sorted(positions):
        raise AssertionError("required report sections are out of order")

    print(
        f"Validated {len(citation_keys)} citation keys, {len(labels)} labels, "
        f"and {len(graphics)} PDF figures."
    )


if __name__ == "__main__":
    main()
