#!/usr/bin/env python3
"""
check_gemma_issues.py

For each issue ID in the known view-ops issue list, fetches the issue
title, body, and all comments from the GitHub API and checks whether the
word "Gemma" appears anywhere in them.

Authentication:
    Set GITHUB_TOKEN in your environment (a classic PAT or fine-grained
    token with at least "read:org" / public-repo read access is enough):

        export GITHUB_TOKEN=ghp_...

    Without a token the script falls back to unauthenticated requests
    (60 req/hour limit — may be enough for this list, but token is safer).

Usage:
    python3 check_gemma_issues.py
"""

import os
import re
import sys
import time
from pathlib import Path

try:
    import requests
except ImportError:
    print("requests not found. Installing...")
    os.system(f"{sys.executable} -m pip install requests -q")
    import requests

try:
    import openpyxl
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils import get_column_letter
except ImportError:
    print("openpyxl not found. Installing...")
    os.system(f"{sys.executable} -m pip install openpyxl -q")
    import openpyxl
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils import get_column_letter

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

REPO        = "torch-spyre/torch-spyre"
API_BASE    = "https://api.github.com"
TOKEN       = os.environ.get("GITHUB_TOKEN", "")
SEARCH_WORD = "gemma"          # case-insensitive
OUTPUT_FILE = Path(__file__).parent / "gemma_issue_matches.xlsx"

# Rate-limit: be polite — 0.5 s between requests when authenticated,
# 1.5 s when unauthenticated (to stay well under the 60/hr cap).
SLEEP_BETWEEN = 0.5 if TOKEN else 1.5

# ---------------------------------------------------------------------------
# All issue IDs from the full view-ops issue list (open + closed)
# ---------------------------------------------------------------------------
ISSUES = [
     46,  122,  230,  265,  266,  267,  287,  310,  312,  328,  342,  344,
    361,  363,  379,  391,  392,  393,  523,  524,  543,  563,  573,  606,
    613,  632,  648,  650,  652,  653,  673,  674,  675,  676,  677,  678,
    680,  684,  688,  689,  690,  691,  694,  697,  700,  720,  721,  722,
    723,  724,  725,  726,  727,  728,  729,  731,  873,  894,  896,  911,
    958, 1077, 1088, 1089, 1090, 1094, 1097, 1110, 1111, 1117, 1120, 1121,
   1149, 1208, 1227, 1279, 1291, 1295, 1333, 1340, 1353, 1369, 1370, 1374,
   1375, 1384, 1395, 1397, 1398, 1405, 1406, 1407, 1413, 1418, 1456, 1457,
   1461, 1462, 1466, 1467, 1470, 1471, 1495, 1526, 1534, 1548, 1549, 1604,
   1649, 1689, 1697, 1714, 1717, 1719, 1722, 1760, 1781, 1783, 1796, 1800,
   1813, 1850, 1859, 1883, 1884, 1919, 1923, 2015, 2017, 2138, 2147, 2245,
   2278, 2294, 2346, 2356, 2409, 2417, 2453, 2526, 2564, 2587, 2614, 2688,
   2791, 2797, 2837, 2848, 2961, 3003, 3137, 3138, 3212, 3225, 3265, 3287,
   3339, 3417, 3455, 3508, 3512, 3558, 3613, 3624, 3627, 3634, 3826, 3851,
   3856, 3916, 4008, 4033, 4050, 4093, 4254, 4271, 4286, 4292, 4311,
]

# ---------------------------------------------------------------------------
# GitHub API helpers
# ---------------------------------------------------------------------------

def _headers() -> dict:
    h = {"Accept": "application/vnd.github+json",
         "X-GitHub-Api-Version": "2022-11-28"}
    if TOKEN:
        h["Authorization"] = f"Bearer {TOKEN}"
    return h


def _get(url: str, params: dict | None = None) -> dict | list | None:
    """GET a single page; return parsed JSON or None on error."""
    try:
        r = requests.get(url, headers=_headers(), params=params, timeout=20)
    except requests.RequestException as exc:
        print(f"  [network error] {exc}")
        return None
    if r.status_code == 404:
        return None          # issue doesn't exist / not accessible
    if r.status_code == 403:
        print(f"  [403 forbidden / rate-limited] — add GITHUB_TOKEN env var")
        return None
    if r.status_code != 200:
        print(f"  [HTTP {r.status_code}] {url}")
        return None
    return r.json()


def _get_all_pages(url: str) -> list:
    """Follow GitHub pagination and return all items combined."""
    items = []
    page = 1
    while True:
        data = _get(url, params={"per_page": 100, "page": page})
        if not data:
            break
        if isinstance(data, list):
            items.extend(data)
            if len(data) < 100:
                break
        else:
            # single object (shouldn't happen for paginated endpoints)
            items.append(data)
            break
        page += 1
        time.sleep(SLEEP_BETWEEN)
    return items


def fetch_issue(issue_num: int) -> dict | None:
    """Return the issue JSON dict or None."""
    url = f"{API_BASE}/repos/{REPO}/issues/{issue_num}"
    return _get(url)


def fetch_comments(issue_num: int) -> list[dict]:
    """Return all comment dicts for an issue."""
    url = f"{API_BASE}/repos/{REPO}/issues/{issue_num}/comments"
    return _get_all_pages(url)


# ---------------------------------------------------------------------------
# Gemma detection
# ---------------------------------------------------------------------------

GEMMA_RE = re.compile(r"\bgemma\b", re.IGNORECASE)


def _find_gemma_in_text(text: str) -> list[str]:
    """Return a list of short snippets (up to 120 chars) around each match."""
    snippets = []
    for m in GEMMA_RE.finditer(text):
        start = max(0, m.start() - 60)
        end   = min(len(text), m.end() + 60)
        snippet = text[start:end].replace("\n", " ").replace("\r", "").strip()
        snippets.append(f"…{snippet}…")
    return snippets


def check_issue_for_gemma(issue_num: int) -> dict:
    """
    Fetch the issue and all its comments; return a result dict:
      {
        "issue":    int,
        "title":    str,
        "url":      str,
        "state":    str,       # open / closed
        "found":    bool,
        "sources":  list[str], # "title", "body", "comment #N by <user>"
        "snippets": list[str], # matching text fragments
      }
    """
    result = {
        "issue":    issue_num,
        "title":    "",
        "url":      f"https://github.com/{REPO}/issues/{issue_num}",
        "state":    "unknown",
        "found":    False,
        "sources":  [],
        "snippets": [],
        "error":    "",
    }

    issue = fetch_issue(issue_num)
    if issue is None:
        result["error"] = "not found / inaccessible"
        return result

    result["title"] = issue.get("title", "")
    result["state"] = issue.get("state", "unknown")

    # Check title
    if GEMMA_RE.search(result["title"]):
        result["found"] = True
        result["sources"].append("title")
        result["snippets"].extend(_find_gemma_in_text(result["title"]))

    # Check body
    body = issue.get("body") or ""
    body_snippets = _find_gemma_in_text(body)
    if body_snippets:
        result["found"] = True
        result["sources"].append("body")
        result["snippets"].extend(body_snippets)

    time.sleep(SLEEP_BETWEEN)

    # Check comments
    comments = fetch_comments(issue_num)
    for c in comments:
        c_body    = c.get("body") or ""
        c_user    = c.get("user", {}).get("login", "unknown")
        c_num     = c.get("id", "?")
        c_snippets = _find_gemma_in_text(c_body)
        if c_snippets:
            result["found"] = True
            result["sources"].append(f"comment by @{c_user}")
            result["snippets"].extend(c_snippets)

    return result


# ---------------------------------------------------------------------------
# Excel output  (same style as extract_view_op_tests.py)
# ---------------------------------------------------------------------------

HEADERS = [
    "Issue #",
    "Title",
    "State",
    "Gemma Mentioned?",
    "Where (title/body/comment)",
    "Matching Snippets",
    "GitHub URL",
]

HEADER_FILL   = PatternFill("solid", fgColor="1F3864")
HEADER_FONT   = Font(bold=True, color="FFFFFF", size=11)
HIT_FILL      = PatternFill("solid", fgColor="E2EFDA")   # green for matches
MISS_FILL     = PatternFill("solid", fgColor="FCE4D6")   # soft red for no match
ERR_FILL      = PatternFill("solid", fgColor="F2F2F2")   # grey for errors
ALT_FILL      = PatternFill("solid", fgColor="DCE6F1")
THIN          = Side(style="thin", color="CCCCCC")
BORDER        = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
WRAP          = Alignment(wrap_text=True, vertical="top")
COL_WIDTHS    = [10, 55, 10, 18, 35, 80, 52]


def write_excel(results: list[dict], output: Path) -> None:
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Gemma Issue Check"

    # Header row
    for col, hdr in enumerate(HEADERS, start=1):
        cell = ws.cell(row=1, column=col, value=hdr)
        cell.font      = HEADER_FONT
        cell.fill      = HEADER_FILL
        cell.border    = BORDER
        cell.alignment = Alignment(wrap_text=True, vertical="center",
                                   horizontal="center")
    ws.freeze_panes = "A2"
    ws.row_dimensions[1].height = 28

    for row_idx, r in enumerate(results, start=2):
        fill = (HIT_FILL  if r["found"]
                else ERR_FILL if r["error"]
                else (ALT_FILL if row_idx % 2 == 0 else None))

        values = [
            r["issue"],
            r["title"],
            r["state"],
            "YES ✓" if r["found"] else ("ERROR" if r["error"] else "no"),
            "; ".join(r["sources"]) if r["sources"] else r.get("error", ""),
            "\n".join(r["snippets"][:6]),    # cap at 6 snippets per row
            r["url"],
        ]
        for col, val in enumerate(values, start=1):
            cell = ws.cell(row=row_idx, column=col, value=val)
            if fill:
                cell.fill = fill
            cell.border    = BORDER
            cell.alignment = WRAP
        ws.row_dimensions[row_idx].height = max(30, min(15 * len(r["snippets"]) + 15, 120))

    for col_idx, width in enumerate(COL_WIDTHS, start=1):
        ws.column_dimensions[get_column_letter(col_idx)].width = width

    # Summary sheet
    ws2 = wb.create_sheet("Summary")
    total   = len(results)
    hits    = sum(1 for r in results if r["found"])
    errors  = sum(1 for r in results if r["error"])
    misses  = total - hits - errors

    ws2["A1"] = "Metric";            ws2["B1"] = "Count"
    ws2["A2"] = "Total issues checked";  ws2["B2"] = total
    ws2["A3"] = "Mention Gemma";         ws2["B3"] = hits
    ws2["A4"] = "Do NOT mention Gemma";  ws2["B4"] = misses
    ws2["A5"] = "Errors / not found";    ws2["B5"] = errors
    ws2["A7"] = "Issues mentioning Gemma:"
    for i, r in enumerate([x for x in results if x["found"]], start=8):
        ws2.cell(row=i, column=1, value=f"#{r['issue']}")
        ws2.cell(row=i, column=2, value=r["title"])
        ws2.cell(row=i, column=3, value=r["url"])

    for row in ws2.iter_rows():
        for cell in row:
            cell.alignment = Alignment(wrap_text=True, vertical="top")
    ws2.column_dimensions["A"].width = 12
    ws2.column_dimensions["B"].width = 60
    ws2.column_dimensions["C"].width = 55

    wb.save(output)
    print(f"\nResults written to: {output}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not TOKEN:
        print("WARNING: GITHUB_TOKEN not set. Using unauthenticated API "
              "(60 req/hour limit). Set it with:\n"
              "  export GITHUB_TOKEN=ghp_...\n")
    else:
        print(f"Using authenticated GitHub API (token: {TOKEN[:8]}…)\n")

    print(f"Repo:        {REPO}")
    print(f"Issues:      {len(ISSUES)}")
    print(f"Search word: '{SEARCH_WORD}' (case-insensitive)")
    print(f"Output:      {OUTPUT_FILE}\n")
    print("-" * 60)

    results: list[dict] = []
    gemma_issues: list[int] = []

    for i, issue_num in enumerate(ISSUES, start=1):
        print(f"[{i:3d}/{len(ISSUES)}] Checking issue #{issue_num} … ", end="", flush=True)
        result = check_issue_for_gemma(issue_num)

        if result["error"]:
            print(f"ERROR: {result['error']}")
        elif result["found"]:
            print(f"FOUND  — '{result['title'][:60]}' | in: {', '.join(result['sources'])}")
            gemma_issues.append(issue_num)
        else:
            print(f"no     — {result['title'][:60]}")

        results.append(result)

    # Print summary
    print("\n" + "=" * 60)
    print(f"SUMMARY: {len(gemma_issues)} / {len(ISSUES)} issues mention Gemma")
    print("=" * 60)
    if gemma_issues:
        print("\nIssues mentioning Gemma:")
        for r in [x for x in results if x["found"]]:
            print(f"  #{r['issue']:5d}  {r['title']}")
            print(f"           Found in: {', '.join(r['sources'])}")
            for snippet in r["snippets"][:3]:
                print(f"           > {snippet[:100]}")
    else:
        print("\nNo issues mention Gemma.")

    write_excel(results, OUTPUT_FILE)


if __name__ == "__main__":
    main()
