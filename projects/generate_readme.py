#!/usr/bin/env python3
"""Generate README.md (by topic) and BY_YEAR.md (by year) from projects.yaml.

Usage (from the projects/ directory, or anywhere):
    pip install pyyaml
    python generate_readme.py          # rewrite README.md and BY_YEAR.md
    python generate_readme.py --check  # exit 1 if they are out of date (for CI)
"""
import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "projects.yaml"
TOPIC_FILE = "README.md"
YEAR_FILE = "BY_YEAR.md"

BANNER = (
    "<!-- AUTO-GENERATED from projects.yaml by generate_readme.py. "
    "Do not edit by hand: edit projects.yaml and re-run the script. -->"
)


def slug(heading):
    """GitHub-style heading anchor."""
    s = heading.strip().lower()
    s = re.sub(r"[^\w\- ]", "", s)
    return s.replace(" ", "-")


def esc(text):
    """Escape characters that would break bold names, e.g. 'Beyond A*'."""
    return re.sub(r"([*_`\[\]])", r"\\\1", text)


def view_switcher(current):
    topic = "📂 **Topic**" if current == "topic" else f"[📂 Topic]({TOPIC_FILE})"
    year = "📅 **Year**" if current == "year" else f"[📅 Year]({YEAR_FILE})"
    return f"**View by:** {topic} &nbsp;|&nbsp; {year}"


def render_project(p, tags=None):
    links = " ".join(f"[[{label}]]({url})" for label, url in (p.get("links") or {}).items())
    head = f"- **{esc(p['name'])}**" + (f" {links}" if links else "")
    line2 = f"  _{p['blurb']}_" if p.get("blurb") else ""
    if tags:
        line2 += " " + tags
    return head + ("\n" + line2 if line2 else "")


def load():
    data = yaml.safe_load(SOURCE.read_text())
    valid = set()
    for s in data["sections"]:
        valid.add(s["title"])
        for sub in s.get("subsections") or []:
            valid.add(f"{s['title']} / {sub}")
    errors = []
    for i, p in enumerate(data["projects"]):
        p["_order"] = i
        if not re.fullmatch(r"\d{4}-\d{2}", str(p.get("date", ""))):
            errors.append(f"{p.get('name')}: date must be \"YYYY-MM\", got {p.get('date')!r}")
        if not p.get("topics"):
            errors.append(f"{p.get('name')}: needs at least one topic")
        for t in p.get("topics") or []:
            if t not in valid:
                errors.append(f"{p['name']}: unknown topic {t!r} (add it to `sections` or fix the spelling)")
    if errors:
        sys.exit("projects.yaml errors:\n  " + "\n  ".join(errors))
    return data


def newest_first(projects):
    # Newest first; projects from the same month keep their order in projects.yaml.
    return sorted(projects, key=lambda p: (-int(p["date"].replace("-", "")), p["_order"]))


def header(data, current):
    return [
        BANNER,
        f"# {data['title']}",
        "",
        view_switcher(current),
        "",
        data["intro"].strip(),
        "",
    ]


def build_topic_view(data):
    by_topic = defaultdict(list)
    for p in data["projects"]:
        for t in p["topics"]:
            by_topic[t].append(p)

    out = header(data, "topic")
    out.append("**Jump to:** " + " · ".join(f"[{s['title']}](#{slug(s['title'])})" for s in data["sections"]))
    for s in data["sections"]:
        out += ["", f"## {s['title']}"]
        groups = [(None, s["title"])] + [(sub, f"{s['title']} / {sub}") for sub in s.get("subsections") or []]
        for sub, key in groups:
            items = newest_first(by_topic.get(key, []))
            if not items:
                continue
            if sub:
                out += ["", f"#### _{sub}_"]
            for p in items:
                out += ["", render_project(p)]
    return "\n".join(out) + "\n"


def build_year_view(data):
    by_year = defaultdict(list)
    for p in data["projects"]:
        by_year[p["date"][:4]].append(p)
    years = sorted(by_year, reverse=True)

    out = header(data, "year")
    out.append("**Jump to:** " + " · ".join(f"[{y}](#{y}) ({len(by_year[y])})" for y in years))
    for y in years:
        out += ["", f"## {y}"]
        for p in newest_first(by_year[y]):
            sections = list(dict.fromkeys(t.split(" / ")[0] for t in p["topics"]))
            tags = "<sub>" + " · ".join(f"[{s}]({TOPIC_FILE}#{slug(s)})" for s in sections) + "</sub>"
            out += ["", render_project(p, tags)]
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="fail if generated files are out of date")
    args = ap.parse_args()

    data = load()
    outputs = {TOPIC_FILE: build_topic_view(data), YEAR_FILE: build_year_view(data)}
    stale = []
    for name, text in outputs.items():
        path = HERE / name
        if path.exists() and path.read_text() == text:
            continue
        if args.check:
            stale.append(name)
        else:
            path.write_text(text)
            print(f"wrote {name}")
    if stale:
        sys.exit(f"Out of date: {', '.join(stale)}. Run `python projects/generate_readme.py` and commit.")


if __name__ == "__main__":
    main()
