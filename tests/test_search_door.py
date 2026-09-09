"""The search door — question-form tutorial titles, robots.txt, and the sitemap the
deployment builds. Agents search the way people ask, so the title has to be the question."""
import pathlib
import re
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
TITLE = re.compile(r"<title>(.*?)</title>", re.S)
OG = re.compile(r'property="og:title" content="([^"]*)"')


def _tutorials(lang):
    d = SITE / "tutorials" if lang == "en" else SITE / "ko" / "tutorials"
    return sorted(p for p in d.glob("*.html") if p.name != "index.html")


@pytest.mark.parametrize("lang", ["en", "ko"])
def test_every_tutorial_title_is_a_question_and_names_the_browser(lang):
    """A title of `4 · Image classifier` is a table of contents. An agent asks *how do I train
    an image classifier in the browser with the PyTorch API* — so that is the title."""
    bad = []
    for p in _tutorials(lang):
        text = p.read_text(encoding="utf-8")
        title = TITLE.search(text).group(1).strip()
        og = OG.search(text)
        if "?" not in title or "브라우저" not in title and "browser" not in title.lower():
            bad.append(f"{p.name}: {title!r}")
        if og is None or og.group(1).strip() != title:
            bad.append(f"{p.name}: og:title {og.group(1) if og else None!r} != <title>")
        if not re.match(r"^\d+ · ", title):
            bad.append(f"{p.name}: the number prefix keeps it in step with the index — {title!r}")
    assert not bad, "\n".join(bad)


def test_the_two_languages_ask_the_same_number_of_questions():
    en, ko = _tutorials("en"), _tutorials("ko")
    assert [p.name for p in en] == [p.name for p in ko]


def test_robots_txt_invites_the_agents_crawlers_and_names_the_sitemap():
    text = (ROOT / "robots.txt").read_text(encoding="utf-8")
    assert "Disallow" not in text, "nothing here is hidden from a crawler"
    for bot in ("GPTBot", "ClaudeBot", "PerplexityBot", "Google-Extended", "CCBot"):
        assert f"User-agent: {bot}" in text, f"{bot} is not named"
    assert re.search(r"^Sitemap: https://playidea-lab\.github\.io/borch/sitemap\.xml$", text, re.M)


def test_build_sitemap_lists_every_tutorial_lesson_and_agent_document(tmp_path, monkeypatch):
    sys.path.insert(0, str(ROOT / "site"))
    import build_sitemap  # noqa: PLC0415
    monkeypatch.setattr(build_sitemap, "OUT", tmp_path / "sitemap.xml")
    assert build_sitemap.main() == 0
    xml = (tmp_path / "sitemap.xml").read_text(encoding="utf-8")
    for p in _tutorials("en") + _tutorials("ko") + sorted((SITE / "learn").glob("*.html")):
        assert f"/{p.relative_to(ROOT).as_posix()}</loc>" in xml, f"{p.name} is not in the sitemap"
    for doc in ("llms.txt", "AGENTS.md", "docs/BOOK.md"):
        assert f"/{doc}</loc>" in xml
    assert "/site/lab/doc/" not in xml, "a built folder's internal pages are not this site's"
    if (SITE / "lab" / "index.html").exists():  # built by site/build_lab.py, absent from a plain checkout
        assert "site/lab/index.html</loc>" in xml, "the built folder's entry is a page of this site"


def test_pages_yml_ships_robots_and_the_sitemap():
    text = (ROOT / ".github" / "workflows" / "pages.yml").read_text(encoding="utf-8")
    assert "python3 site/build_sitemap.py" in text
    gather = text[text.index("gather what goes up"):text.index("upload-pages-artifact")]
    assert "robots.txt" in gather and "sitemap.xml" in gather


def test_the_search_console_file_is_at_the_root_and_the_deployment_copies_it():
    """Search Console re-checks the file it verified with; losing it un-verifies the property,
    and the loss is silent — the console simply stops reporting one day."""
    files = sorted(ROOT.glob("google*.html"))
    assert len(files) == 1, f"expected exactly one Search Console file at the root, found {files}"
    name = files[0].name
    assert files[0].read_text(encoding="utf-8").strip() == f"google-site-verification: {name}"
    gather = (ROOT / ".github" / "workflows" / "pages.yml").read_text(encoding="utf-8")
    gather = gather[gather.index("gather what goes up"):gather.index("upload-pages-artifact")]
    assert "cp google*.html _site/" in gather, "the verification file is not copied into the deployment"


def test_the_sitemap_does_not_advertise_the_verification_file():
    """It is a token for one reader, not a page of this site."""
    sys.path.insert(0, str(ROOT / "site"))
    import build_sitemap  # noqa: PLC0415
    assert not any(p.startswith("google") for p in build_sitemap.pages() + list(build_sitemap.DOCUMENTS))
