"""The integrated-GPU hint on Windows — one string in two languages, one anchor on two pages,
and the landing page's code that ties them. Measured 2026-09-09: a notebook with an RTX 5050
answered `intel / gen-12lp`, and nothing on the page said what to do about it."""
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SITE = ROOT / "site"


def test_the_hint_string_exists_in_both_languages_and_names_the_setting():
    js = (SITE / "assets" / "i18n.js").read_text(encoding="utf-8")
    block = js[js.index('"device.integratedOnWindows"'):js.index('"run.done"')]
    assert re.search(r"en:.*Graphics", block, re.S) and "chrome.exe" in block and "High performance" in block
    assert re.search(r"ko:.*그래픽", block, re.S) and "고성능" in block


def test_the_landing_page_shows_the_hint_for_an_intel_adapter_on_windows_only():
    home = (SITE / "assets" / "home.js").read_text(encoding="utf-8")
    assert "const ON_WINDOWS" in home
    assert 't("device.integratedOnWindows")' in home
    assert "#windows-laptop" in home, "the hint has to point at the setup page's steps"
    assert re.search(r"ON_WINDOWS && /\\bintel\\b/i\.test", home), "Intel on Windows, and nothing else, gets the hint"
    assert "arc" in home, "an Intel Arc is a discrete card and must not be told it is integrated"


@pytest.mark.parametrize("name", ["setup.html", "ko/setup.html"])
def test_the_setup_page_has_the_steps_under_the_anchor_the_hint_links_to(name):
    page = (SITE / name).read_text(encoding="utf-8")
    assert 'id="windows-laptop"' in page
    section = page[page.index('id="windows-laptop"'):]
    section = section[:section.index("<h2")] if "<h2" in section else section
    assert "chrome.exe" in section and "intel / gen-12lp" in section
    assert ("High performance" in section) or ("고성능" in section)
