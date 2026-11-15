"""Every word the Workbench shows comes from copy_deck.toml, and nothing else slips in."""

import html
import re
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent / "strawberrywatch" / "workbench"
DECK = tomllib.loads((ROOT / "copy_deck.toml").read_text())
SECTIONS = {name: s for name, s in DECK.items() if isinstance(s, dict) and "file" in s}
ALL_STRINGS = sorted(
    {s for sec in SECTIONS.values() for k in ("strings", "new_strings") for s in sec.get(k, [])},
    key=len,
    reverse=True,
)


def collapse(value):
    return re.sub(r"\s+", " ", value).strip()


def visible(path):
    source = path.read_text()
    if path.suffix != ".html":
        return collapse(source)
    source = re.sub(r"\{#.*?#\}", " ", source, flags=re.S)
    source = re.sub(r"\{%.*?%\}", " ", source, flags=re.S)
    source = re.sub(r"<[^>]+>", " ", source)
    return collapse(html.unescape(source))


@pytest.mark.parametrize("name", sorted(SECTIONS))
def test_every_deck_string_is_in_its_file(name):
    section = SECTIONS[name]
    path = ROOT / section["file"]
    text = visible(path)
    raw = collapse(path.read_text())
    for string in section.get("strings", []) + section.get("new_strings", []):
        assert collapse(string) in text, f"{section['file']} is missing {string!r}"
    for attr in section.get("attrs", []):
        assert collapse(attr) in raw, f"{section['file']} is missing {attr!r}"


@pytest.mark.parametrize("path", sorted((ROOT / "templates").glob("*.html")), ids=lambda p: p.name)
def test_templates_show_no_words_outside_the_deck(path):
    text = visible(path)
    for string in ALL_STRINGS:
        text = text.replace(collapse(string), " ")
    text = re.sub(r"\{\{.*?\}\}", " ", text)
    leftover = re.sub(r"[\s\d.,:;!?'\"()&/\-]", "", text)
    assert leftover == "", f"{path.name} shows {text.strip()!r}"


def test_no_em_dashes_anywhere():
    for path in ROOT.rglob("*"):
        if path.is_file() and path.suffix in {".py", ".html", ".css", ".js", ".md", ".toml"}:
            assert "—" not in path.read_text(), path.name
