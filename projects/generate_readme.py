#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Generate the project lists, by topic and by year, from projects.yaml:

  projects/README.md and projects/BY_YEAR.md, shown on GitHub, and
  gh-pages/README.md and gh-pages/BY_YEAR.md, the same two views on the website: its home page,
  https://facebookresearch.github.io/RAM/, and https://facebookresearch.github.io/RAM/by-year/

Usage (from the projects/ directory, or anywhere):
    pip install pyyaml
    python generate_readme.py          # rewrite all four files
    python generate_readme.py --check  # exit 1 if any of them are out of date (for CI)
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from urllib.parse import urljoin

import yaml

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "projects.yaml"
SITE_DIR = HERE.parent / "gh-pages"  # Jekyll source of the website
TOPIC_FILE = "README.md"
YEAR_FILE = "BY_YEAR.md"
# Relative links in projects.yaml point into projects/, which is not part of the website.
PROJECTS_ON_GITHUB = "https://github.com/facebookresearch/RAM/tree/main/projects/"

BANNER = (
    "<!-- AUTO-GENERATED from {dir}projects.yaml by {dir}generate_readme.py. "
    "Do not edit by hand: edit {dir}projects.yaml and re-run the script. -->"
)


def slug(heading):
    """GitHub-style heading anchor."""
    s = heading.strip().lower()
    s = re.sub(r"[^\w\- ]", "", s)
    return s.replace(" ", "-")


def esc(text):
    """Escape characters that would break bold names, e.g. 'Beyond A*'."""
    return re.sub(r"([*_`\[\]])", r"\\\1", text)


def view_switcher(current, site):
    topic = "📂 **Topic**" if current == "topic" else f"[📂 Topic]({TOPIC_FILE})"
    year = "📅 **Year**" if current == "year" else f"[📅 Year]({YEAR_FILE})"
    # The website's markdown renderer (kramdown) turns a line with a bare "|" into a table.
    sep = "\\|" if site else "|"
    return f"**View by:** {topic} &nbsp;{sep}&nbsp; {year}"


def render_project(p, site, tags=None):
    links = " ".join(
        f"[[{label}]]({urljoin(PROJECTS_ON_GITHUB, url) if site else url})"
        for label, url in (p.get("links") or {}).items()
    )
    head = f"- **{esc(p['name'])}**" + (f" {links}" if links else "")
    line2 = ""
    if p.get("blurb"):
        # kramdown does not nest emphasis: in _..*word*.._ the inner asterisks would show.
        line2 = f"  <em>{p['blurb']}</em>" if site else f"  _{p['blurb']}_"
    if tags:
        line2 += " " + tags
    return head + ("\n" + line2 if line2 else "")


def load():
    data = yaml.safe_load(SOURCE.read_text(encoding="utf-8"))
    valid = set()
    for s in data["sections"]:
        valid.add(s["title"])
        for sub in s.get("subsections") or []:
            valid.add(f"{s['title']} / {sub}")
    errors = []
    for i, p in enumerate(data["projects"]):
        p["_order"] = i
        if not re.fullmatch(r"\d{4}-\d{2}", str(p.get("date", ""))):
            errors.append(
                f"{p.get('name')}: date must be \"YYYY-MM\", got {p.get('date')!r}"
            )
        if not p.get("topics"):
            errors.append(f"{p.get('name')}: needs at least one topic")
        for t in p.get("topics") or []:
            if t not in valid:
                errors.append(
                    f"{p['name']}: unknown topic {t!r} (add it to `sections` or fix the spelling)"
                )
    if errors:
        sys.exit("projects.yaml errors:\n  " + "\n  ".join(errors))
    return data


def newest_first(projects):
    # Newest first; projects from the same month keep their order in projects.yaml.
    return sorted(
        projects, key=lambda p: (-int(p["date"].replace("-", "")), p["_order"])
    )


def header(data, current, site):
    page = data["website"] if site else data
    out = []
    if site:  # Jekyll front matter: the page's browser-tab title and URL
        if current == "topic":
            title, url = page["title"], "/"
        else:
            title, url = "Projects by year", "/by-year/"
        title = json.dumps(title, ensure_ascii=False)
        out += ["---", f"title: {title}", f"permalink: {url}", "---"]
    return out + [
        BANNER.format(dir="projects/" if site else ""),
        f"# {page['title']}",
        "",
        view_switcher(current, site),
        "",
        page["intro"].strip(),
        "",
    ]


def build_topic_view(data, site):
    by_topic = defaultdict(list)
    for p in data["projects"]:
        for t in p["topics"]:
            by_topic[t].append(p)

    out = header(data, "topic", site)
    out.append(
        "**Jump to:** "
        + " · ".join(f"[{s['title']}](#{slug(s['title'])})" for s in data["sections"])
    )
    for s in data["sections"]:
        out += ["", f"## {s['title']}"]
        groups = [(None, s["title"])] + [
            (sub, f"{s['title']} / {sub}") for sub in s.get("subsections") or []
        ]
        for sub, key in groups:
            items = newest_first(by_topic.get(key, []))
            if not items:
                continue
            if sub:
                out += ["", f"#### _{sub}_"]
            for p in items:
                out += ["", render_project(p, site)]
    return "\n".join(out) + "\n"


def build_year_view(data, site):
    by_year = defaultdict(list)
    for p in data["projects"]:
        by_year[p["date"][:4]].append(p)
    years = sorted(by_year, reverse=True)

    out = header(data, "year", site)
    out.append(
        "**Jump to:** " + " · ".join(f"[{y}](#{y}) ({len(by_year[y])})" for y in years)
    )
    for y in years:
        out += ["", f"## {y}"]
        for p in newest_first(by_year[y]):
            sections = list(dict.fromkeys(t.split(" / ")[0] for t in p["topics"]))
            tags = (
                "<sub>"
                + " · ".join(f"[{s}]({TOPIC_FILE}#{slug(s)})" for s in sections)
                + "</sub>"
            )
            out += ["", render_project(p, site, tags)]
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--check", action="store_true", help="fail if generated files are out of date"
    )
    args = ap.parse_args()

    data = load()
    outputs = {}
    for out_dir, site in ((HERE, False), (SITE_DIR, True)):
        outputs[out_dir / TOPIC_FILE] = build_topic_view(data, site)
        outputs[out_dir / YEAR_FILE] = build_year_view(data, site)
    stale = []
    for path, text in outputs.items():
        name = path.relative_to(HERE.parent).as_posix()
        if path.exists() and path.read_text(encoding="utf-8") == text:
            continue
        if args.check:
            stale.append(name)
        else:
            path.write_text(text, encoding="utf-8")
            print(f"wrote {name}")
    if stale:
        sys.exit(
            f"Out of date: {', '.join(stale)}. Run `python projects/generate_readme.py` and commit."
        )


if __name__ == "__main__":
    main()
