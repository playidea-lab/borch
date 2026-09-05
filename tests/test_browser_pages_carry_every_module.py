"""The pages that copy `borch` and `borch_webgpu` into Pyodide **by module name** name
every module the packages have.

Three pages under `tests/browser/` build the packages inside Pyodide from a list of file
names (`const PACKAGES = {...}`) rather than from a wheel — they are the checks that run
the *source tree*, so a wheel is the wrong thing for them to install. The price is a list
that has to follow the directory. It did not: `_hub` and `_report` were added to
`borch_webgpu`, two of the three pages were updated, and `onnx_binding.html` was not. Its
`__init__` then imported a module that had not been copied and Pyodide said *cannot import
name '_hub' from partially initialized module* — the wording of a circular import, for a
missing file. Nothing caught it until the nightly grew a row for that page (2026-09-06).

So the lists are read here and held to the directories. A module that arrives lands in
this test the same day; a page that falls behind is named.
"""
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
PAGES = ("onnx_binding.html", "scope_escape.html", "runner.html")
PACKAGES = ("borch", "borch_webgpu")


def _listed(page: pathlib.Path, package: str) -> set[str]:
    text = page.read_text(encoding="utf-8")
    block = re.search(r"const PACKAGES = \{(.*?)\};", text, re.S)
    assert block, f"{page.name}: no PACKAGES block"
    entry = re.search(package + r":\s*\[(.*?)\]", block.group(1), re.S)
    assert entry, f"{page.name}: PACKAGES has no {package}"
    return set(re.findall(r'"([^"]+)"', entry.group(1)))


def _present(package: str) -> set[str]:
    return {p.stem for p in (ROOT / package).glob("*.py")}


def test_every_page_that_copies_the_packages_names_every_module():
    gaps = []
    for name in PAGES:
        page = ROOT / "tests" / "browser" / name
        for package in PACKAGES:
            listed, present = _listed(page, package), _present(package)
            missing, extra = sorted(present - listed), sorted(listed - present)
            if missing:
                gaps.append(f"{name}: {package} is missing {', '.join(missing)}")
            if extra:
                gaps.append(f"{name}: {package} lists {', '.join(extra)}, which the directory does not have")
    assert not gaps, "a page's module list does not match the package:\n  " + "\n  ".join(gaps)
