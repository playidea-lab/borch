"""The agent door — `AGENTS.md`, `llms.txt`, `site/build_llms.py` and the two manifests.

An agent reads these instead of the repository, so a link that 404s or a code block that
no longer runs is a wrong answer handed to every agent at once. The checks read the
documents rather than repeating their contents here.
"""
import json
import pathlib
import re
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
PAGES = "https://playidea-lab.github.io/borch/"
# Deployed by `pages.yml` but absent from a checkout: the build makes them.
BUILT = {"llms-full.txt", "site/assets/api.json", "site/assets/api-index.json", "site/lab/", "site/marimo/"}
LINK = re.compile(r"\]\(([^)\s]+)\)")
FENCE = re.compile(r"```(\w+)\n(.*?)```", re.S)


def _links(path):
    return LINK.findall((ROOT / path).read_text(encoding="utf-8"))


def _repo_path(url):
    """A Pages URL or a relative link, as the path it names in this checkout — or None for a foreign host."""
    if url.startswith(PAGES):
        return url[len(PAGES):].split("#")[0]
    if url.startswith(("http://", "https://")):
        return None
    return url.split("#")[0]


@pytest.mark.parametrize("doc", ["llms.txt", "AGENTS.md"])
def test_every_link_names_a_file_in_the_checkout_or_one_the_build_makes(doc):
    missing = []
    for url in _links(doc):
        rel = _repo_path(url)
        if rel is None or rel == "":
            continue
        if rel in BUILT or any(rel.startswith(b) for b in BUILT if b.endswith("/")):
            continue
        target = ROOT / rel
        if rel.endswith("/"):
            target = target / "index.html"
        if not target.exists():
            missing.append(url)
    assert not missing, f"{doc} links to nothing at: {missing}"


def _gathered_by_pages():
    """The names `pages.yml`'s gather step copies into `_site` — the deployment is that list, not the checkout."""
    text = (ROOT / ".github" / "workflows" / "pages.yml").read_text(encoding="utf-8")
    block = text[text.index("gather what goes up"):text.index("upload-pages-artifact")]
    names = set()
    for line in block.splitlines():
        line = line.strip()
        if line.startswith("cp ") and not line.startswith("cp -r"):
            names.update(line.split()[1:-1])
        elif line.startswith("cp -r "):
            names.update(line.split()[2:-1])
        elif line.startswith("mkdir -p _site/") and "&& cp " in line:
            names.add(line.split("&& cp ")[1].split()[0])
    return names


@pytest.mark.parametrize("doc", ["llms.txt", "AGENTS.md"])
def test_every_pages_url_names_something_the_deployment_actually_copies(doc):
    """The first deployment after llms.txt was written answered 404 to README, AGENTS.md, the
    book, BORCH-TS.md and ROADMAP.md: Pages ships what the gather step lists, and the link
    check above only asked whether the file exists in the checkout."""
    gathered = _gathered_by_pages()
    not_shipped = []
    for url in _links(doc):
        if not url.startswith(PAGES):
            continue
        rel = url[len(PAGES):].split("#")[0]
        if rel == "" or rel in BUILT:
            continue
        top = rel.split("/")[0]
        if rel in gathered or top in gathered or f"{top}/" in {g.split("/")[0] + "/" for g in gathered}:
            continue
        if any(rel.startswith(g.rstrip("/") + "/") for g in gathered):
            continue
        not_shipped.append(url)
    assert not not_shipped, f"{doc} links to Pages URLs the gather step in pages.yml never copies: {not_shipped}"


def test_llms_txt_lists_every_tutorial_and_no_page_that_is_not_there():
    listed = {l.rsplit("/", 1)[1] for l in _links("llms.txt") if "/site/tutorials/" in l and l.endswith(".html")}
    on_disk = {p.name for p in (ROOT / "site" / "tutorials").glob("*.html")} - {"index.html"}
    assert listed == on_disk, f"llms.txt tutorials {sorted(listed ^ on_disk)} disagree with site/tutorials/"


def test_the_python_smoke_tests_in_agents_md_run_on_the_numpy_core():
    sys.path.insert(0, str(ROOT))
    ran = 0
    for lang, body in FENCE.findall((ROOT / "AGENTS.md").read_text(encoding="utf-8")):
        if lang != "python" or body.lstrip().startswith("%pip") or "borch_webgpu" in body:
            continue  # the notebook cell needs Pyodide and a GPU; the browser checks cover it
        exec(compile(body, "AGENTS.md", "exec"), {})  # noqa: S102 — the document's own block
        ran += 1
    assert ran >= 1, "AGENTS.md has no Python block the core can run — the smoke test is gone"


def test_the_two_manifests_carry_the_same_keywords():
    npm = json.loads((ROOT / "package.json").read_text())["keywords"]
    toml = (ROOT / "pyproject.toml").read_text()
    m = re.search(r"^keywords = (\[.*\])$", toml, re.M)
    assert m, "pyproject.toml has no keywords line"
    assert set(npm) == set(json.loads(m.group(1))), "npm and PyPI would be found by different words"
    assert {"pytorch", "webgpu", "pyodide"} <= set(npm)


def test_build_llms_concatenates_the_documents_llms_txt_names(tmp_path, monkeypatch):
    sys.path.insert(0, str(ROOT / "site"))
    import build_llms  # noqa: PLC0415
    monkeypatch.setattr(build_llms, "OUT", tmp_path / "llms-full.txt")
    assert build_llms.main() == 0
    text = (tmp_path / "llms-full.txt").read_text(encoding="utf-8")
    for rel in build_llms.PARTS:
        assert f"===== {rel} =====" in text, f"{rel} is not in llms-full.txt"
    assert text.count("=====") == 2 * len(build_llms.PARTS)


def test_context7_json_is_valid_and_its_rules_are_the_agents_md_rules_in_short():
    """`context7.json` is what Context7 hands an agent verbatim, so its rules are AGENTS.md's ten
    in one line each. The file has to parse, exclude the built folders, and carry every rule."""
    cfg = json.loads((ROOT / "context7.json").read_text(encoding="utf-8"))
    assert cfg["$schema"].endswith("/context7.json")
    assert 8 <= len(cfg["rules"]) <= 15
    assert all(len(r) < 500 and "\n" not in r for r in cfg["rules"]), "a rule is a line, not a page"
    for built in ("node_modules", "vendor", "site/lab", "site/marimo", "borch-ts/dist"):
        assert built in cfg["excludeFolders"], f"{built} would be indexed as documentation"
    agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    for needle in ("borch.install(\"borch\")", "api-index.json", "torch.compile", "using s = scope()", "complex64", "exportOnnx"):
        assert needle in agents and any(needle in r for r in cfg["rules"]), f"{needle!r} is in one document and not the other"


def test_the_support_check_snippet_in_agents_md_reads_the_index_and_names_the_absent(monkeypatch, capsys):
    """The snippet fetches api-index.json; here it is handed the checkout's copy. Names the
    range promises have to resolve, and the two that are absent by design have to say so."""
    import io
    import urllib.request
    local = ROOT / "site" / "assets" / "api-index.json"
    if not local.exists():
        pytest.skip("site/assets/api-index.json is built by site/build_api.py")
    monkeypatch.setattr(urllib.request, "urlopen", lambda url: io.BytesIO(local.read_bytes()))
    blocks = [b for lang, b in FENCE.findall((ROOT / "AGENTS.md").read_text(encoding="utf-8")) if lang == "python" and "api-index.json" in b]
    assert len(blocks) == 1
    exec(compile(blocks[0], "AGENTS.md", "exec"), {})  # noqa: S102
    out = capsys.readouterr().out
    for name in ("AdamW", "LayerNorm", "MultiheadAttention", "stft"):
        assert re.search(rf"^{name}\s+\S", out, re.M) and "not in borch" not in re.search(rf"^{name}.*$", out, re.M).group(0), f"{name} should resolve"
    for name in ("autocast", "compile"):
        assert re.search(rf"^{name}\s+— not in borch", out, re.M), f"{name} is absent by design and must say so"
