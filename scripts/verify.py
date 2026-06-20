#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
verify.py — Run all code quality checks for AgentRubric.

Usage:
    python scripts/verify.py

Exits with code 0 if all checks pass.
Exits with code 1 if any check fails.
Run this after making changes to confirm nothing regressed.
"""

import sys
import re
from pathlib import Path

ROOT = Path(__file__).parent.parent
PKG  = ROOT / "agentrubric"


def run_grep(label: str, patterns: list[str], paths: list[str],
             expect_empty: bool = True) -> bool:
    """Search files for patterns using Python (cross-platform alternative to grep).

    Args:
        label: Human-readable check name for output.
        patterns: List of regex patterns to search for.
        paths: List of file or directory paths to search.
        expect_empty: If True, pass when no matches found.
                      If False, pass when matches are found.

    Returns:
        True if check passed, False if failed.
    """
    import re
    from pathlib import Path

    combined_pattern = "|".join(f"({p})" for p in patterns)
    regex = re.compile(combined_pattern)
    matches = []

    for path_str in paths:
        path = Path(path_str)
        if path.is_file():
            files = [path]
        elif path.is_dir():
            files = list(path.glob("**/*.py"))
        else:
            continue

        for file in files:
            try:
                with open(file, 'r', encoding='utf-8', errors='ignore') as f:
                    for line_num, line in enumerate(f, 1):
                        if regex.search(line):
                            matches.append(f"{file}:{line_num}: {line.rstrip()}")
            except Exception:
                pass

    output = "\n".join(matches[:3])  # Show first 3 matches

    if expect_empty:
        passed = len(matches) == 0
    else:
        passed = len(matches) > 0

    status = "[PASS]" if passed else "[FAIL]"
    print(f"  {status}  {label}")
    if not passed and expect_empty and matches:
        for match in matches[:3]:
            # Encode safely to handle Windows console encoding issues
            try:
                print(f"     {match}")
            except UnicodeEncodeError:
                print(f"     {match.encode('utf-8', errors='replace').decode('utf-8', errors='replace')}")
    return passed


def check_file_exists(label: str, path: Path) -> bool:
    """Check that a file exists.

    Args:
        label: Human-readable check name.
        path: Path to check.

    Returns:
        True if file exists, False otherwise.
    """
    passed = path.exists()
    status = "[PASS]" if passed else "[FAIL]"
    print(f"  {status}  {label}")
    if not passed:
        print(f"     Missing: {path}")
    return passed


def main() -> int:
    """Run all checks and return exit code.

    Returns:
        0 if all checks passed, 1 if any failed.
    """
    print()
    print("  AgentRubric — code quality checks")
    print()

    results = []

    # ── Infrastructure files ──────────────────────────────────────────
    print("  Infrastructure:")
    results.append(check_file_exists(
        "constants.py exists", PKG / "constants.py"))
    results.append(check_file_exists(
        "utils.py exists",     PKG / "utils.py"))
    results.append(check_file_exists(
        "logger.py exists",    PKG / "logger.py"))

    print()

    # ── Magic numbers ─────────────────────────────────────────────────
    print("  Magic numbers:")
    results.append(run_grep(
        "No hardcoded divergence thresholds in hack_detector",
        [r"0\.10", r"0\.40"],
        [str(PKG / "agents" / "hack_detector.py")],
        expect_empty=True,
    ))
    results.append(run_grep(
        "No hardcoded pass threshold (0.6) in scorer",
        [r">=\s*0\.6", r"<\s*0\.6"],
        [str(PKG / "rubric_scorer.py")],
        expect_empty=True,
    ))

    print()

    # ── Truncation patterns ───────────────────────────────────────────
    print("  Truncation utilities:")
    results.append(run_grep(
        "No raw truncation patterns in agent files",
        [r"\[:60\]", r"\[:300\]", r"\[:40\]", r"\[:80\]", r"\[:150\]"],
        [
            str(PKG / "agents"),
            str(PKG / "eval"),
            str(PKG / "rubric_scorer.py"),
        ],
        expect_empty=True,
    ))

    print()

    # ── Type hints ────────────────────────────────────────────────────
    print("  Type hints:")
    results.append(run_grep(
        "No Optional[] remains (all replaced with X | None)",
        [r"Optional\["],
        [str(PKG)],
        expect_empty=True,
    ))

    print()

    # ── Logging ───────────────────────────────────────────────────────
    print("  Logging:")
    # Check for print() in data processing code (not in __main__ or display functions)
    print_violations = []
    display_func_prefixes = ("print_", "_print")
    for path_str in [str(PKG / "agents"), str(PKG / "eval"), str(PKG / "flywheel"), str(PKG / "graph")]:
        path = Path(path_str)
        if not path.exists():
            continue
        files = list(path.glob("**/*.py")) if path.is_dir() else [path]
        for file in files:
            try:
                with open(file, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
                    in_main = False
                    in_display_func = False
                    current_func = ""
                    for line_num, line in enumerate(lines, 1):
                        # Track if entering __main__ block
                        if 'if __name__ == "__main__"' in line or "if __name__ == '__main__'" in line:
                            in_main = True
                            continue
                        # Track if exiting __main__ block
                        if in_main and line.strip() and not line.startswith((' ', '\t')):
                            in_main = False
                        # Track current function name
                        if re.match(r"^def (\w+)\(", line):
                            match = re.match(r"^def (\w+)\(", line)
                            current_func = match.group(1) if match else ""
                            in_display_func = any(current_func.startswith(p) for p in display_func_prefixes)
                        # Flag print() in non-display, non-__main__ code
                        if not in_main and not in_display_func and re.search(r"^\s{4,}print\(", line):
                            print_violations.append(f"{file}:{line_num}: {line.rstrip()}")
            except Exception:
                pass

    passed = len(print_violations) == 0
    print(f"  [{'PASS' if passed else 'FAIL'}]  No standalone print() in node functions")
    if print_violations:
        for v in print_violations[:3]:
            print(f"     {v}")
    results.append(passed)

    print()

    # ── Constants usage ───────────────────────────────────────────────
    print("  Constants:")
    results.append(run_grep(
        "constants.py imported in hack_detector",
        ["from agentrubric.constants import"],
        [str(PKG / "agents" / "hack_detector.py")],
        expect_empty=False,
    ))
    results.append(run_grep(
        "constants.py imported in transcript_filter",
        ["from agentrubric.constants import"],
        [str(PKG / "flywheel" / "transcript_filter.py")],
        expect_empty=False,
    ))
    results.append(run_grep(
        "DEFAULT_HACK_THRESHOLD used in graph.py",
        ["DEFAULT_HACK_THRESHOLD"],
        [str(PKG / "graph" / "graph.py")],
        expect_empty=False,
    ))

    print()

    # ── Input validation ─────────────────────────────────────────────
    print("  Input validation:")
    results.append(run_grep(
        "_validate_inputs defined and called in graph.py",
        ["_validate_inputs"],
        [str(PKG / "graph" / "graph.py")],
        expect_empty=False,
    ))

    print()

    # ── Refactored pipeline ───────────────────────────────────────────
    print("  Pipeline structure:")
    results.append(run_grep(
        "_process_single_sample exists in run_pipeline.py",
        ["def _process_single_sample"],
        [str(PKG / "run_pipeline.py")],
        expect_empty=False,
    ))
    results.append(run_grep(
        "_format_result exists in run_pipeline.py",
        ["def _format_result"],
        [str(PKG / "run_pipeline.py")],
        expect_empty=False,
    ))
    results.append(run_grep(
        "_categorise_error exists in run_pipeline.py",
        ["def _categorise_error"],
        [str(PKG / "run_pipeline.py")],
        expect_empty=False,
    ))

    print()

    # ── Summary ───────────────────────────────────────────────────────
    passed = sum(results)
    total  = len(results)
    failed = total - passed

    if failed == 0:
        print(f"  All {total} checks passed.")
    else:
        print(f"  {failed}/{total} checks failed.")

    print()
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
