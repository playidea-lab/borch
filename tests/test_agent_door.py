"""The agent door — `AGENTS.md`, `llms.txt`, `site/build_llms.py` and the two manifests.

An agent reads these instead of the repository, so a link that 404s or a code block that
no longer runs is a wrong answer handed to every agent at once. The checks read the
documents rather than repeating their contents here.
"""
import json
import pathlib
import re
import sys
import textwrap

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
            # **Nothing runs these.** The line here used to say the browser checks cover
            # them and no probe reads this file — measured 2026-09-15. What holds them is
            # the test below, which checks the names they use are names that exist.
            continue
        exec(compile(textwrap.dedent(body), "AGENTS.md", "exec"), {})  # noqa: S102 — the document's own block
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
    # Context7 validates on submission: description at most 200 characters, a rule at most 255.
    # The first submission (2026-09-08) was accepted with the file rejected — two rules and the
    # description over the limit, so the agent would have got no rules at all.
    assert len(cfg["description"]) <= 200, f"description is {len(cfg['description'])} chars; Context7 allows 200"
    long = [(i, len(r)) for i, r in enumerate(cfg["rules"]) if len(r) > 255]
    assert not long, f"rules over Context7's 255 characters: {long}"
    assert all("\n" not in r for r in cfg["rules"]), "a rule is a line, not a page"
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
    exec(compile(textwrap.dedent(blocks[0]), "AGENTS.md", "exec"), {})  # noqa: S102
    out = capsys.readouterr().out
    for name in ("AdamW", "LayerNorm", "MultiheadAttention", "stft"):
        assert re.search(rf"^{name}\s+\S", out, re.M) and "not in borch" not in re.search(rf"^{name}.*$", out, re.M).group(0), f"{name} should resolve"
    for name in ("autocast", "compile"):
        assert re.search(rf"^{name}\s+— not in borch", out, re.M), f"{name} is absent by design and must say so"


def test_the_documented_cdn_url_is_the_published_line_of_this_package():
    """The documents hand a reader a CDN URL. It carries a version range, and a range that
    no longer covers this package sends them to the previous minor without saying so."""
    version = json.loads((ROOT / "package.json").read_text())["version"]
    major_minor = ".".join(version.split(".")[:2])
    want = f"borch-ts@{major_minor}/+esm"
    for doc in ("README.md", "AGENTS.md", "skills/borch/SKILL.md"):
        text = (ROOT / doc).read_text(encoding="utf-8")
        assert want in text, f"{doc} does not offer {want} — borch-ts is {version} now"
    assert "cdn:py" in json.loads((ROOT / "package.json").read_text())["scripts"], (
        "the CDN entry point is documented and nothing runs it")


def test_both_registry_summaries_say_what_only_this_library_claims():
    """**The summary is the line a search result shows, and it is where a reader decides.**
    Measured 2026-09-10: asked for a pure-Python package that prints the same values and the
    same error messages as torch — which is this library's own sentence — an agent searched,
    found nothing, and answered that no such package exists. Neither summary said "error
    messages"; both led with the browser, which is not what that question asks about."""
    toml = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    npm = json.loads((ROOT / "package.json").read_text())["description"]
    py = re.search(r'^description = "(.*)"$', toml, re.M).group(1)
    for what, text in (("pyproject.toml", py), ("package.json", npm)):
        assert "error message" in text, (
            f"{what}'s summary does not say it matches torch's error messages, which is the "
            f"claim no competitor makes:\n    {text}")
        assert "torch" in text.lower(), f"{what}'s summary does not name torch"
        assert len(text) <= 300, f"{what}'s summary is {len(text)} chars; PyPI truncates a long one"


WORKBENCH_CALL = re.compile(r"\b(?:wb|torch\.workbench)\.(\w+)")
SESSION_READ = re.compile(r"\bs\.(\w+)")


def test_the_workbench_names_in_agents_md_are_names_that_exist():
    """**The block that cannot be run has to be held some other way.**

    `AGENTS.md`'s workbench example opens `import borch_webgpu`, which needs Pyodide and a
    tab, so the runner above skips it and — measured — no browser probe reads this file
    either. A document nothing checks is a document that goes stale: this repository has
    found that in `parity.ts`, in `platform_claims`, in the census and in `onnx_trap`.

    So the names are checked even though the code is not: every `wb.x` and every `s.x` the
    document reads has to be something the surface offers.
    """
    import borch._workbench as core                                   # noqa: PLC0415

    whole = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    # **This section only.** `s` is a scope elsewhere in the document (`s.keep`), and a
    # search over the whole file reads that as a Session field and fails on it.
    start = whole.index("## One call that trains on a folder of images")
    text = whole[start:whole.index("\n## ", start + 1)]
    binding = (ROOT / "borch_webgpu" / "_workbench.py").read_text(encoding="utf-8")
    offered = {m.group(1) for m in re.finditer(r"^def (\w+)", binding, re.M)}
    offered |= {n for n in dir(core) if not n.startswith("_")}
    session = {n for n in dir(core.Session) if not n.startswith("_")}

    missing = sorted({n for n in WORKBENCH_CALL.findall(text) if n not in offered})
    assert not missing, (
        "AGENTS.md calls these on the workbench and the surface has no such name:\n  "
        + "\n  ".join(missing))
    # `s` is a Session in that block; the fields it reads have to be fields it has.
    read = {n for n in SESSION_READ.findall(text)} - {"x"}
    absent = sorted(n for n in read if n not in session and n not in _SESSION_FIELDS)
    assert not absent, (
        "AGENTS.md reads these off a Session and it does not carry them:\n  "
        + "\n  ".join(absent))


# Fields a Session sets in `__init__` or `fit` rather than declaring on the class, so
# `dir(Session)` does not show them.
_SESSION_FIELDS = {
    "accuracy", "measured_on", "seconds", "how", "losses", "features", "predicted",
    "model", "head", "config", "data", "masks", "given", "iou", "score", "score_name",
    "held_out", "torch", "k",
}


# The names the document says are outside the index, as it spells them.
OUTSIDE_INDEX = re.compile(r"The names that\nexist only in a browser are not in them: (.+?)\. Those", re.S)


def test_the_names_agents_md_says_are_outside_the_index_are_outside_it_and_real():
    """**An instruction that makes an agent deny something true is worse than none.**

    `AGENTS.md` tells a reader to check `api-index.json` and, for anything absent, to say
    it does not exist. The index is generated from borch-ts's declarations, so every
    browser-only name is absent from it — `hub`, `workbench`, `suspects`, `report` were
    all measured absent, and all four exist. The document lists them; this holds the list
    to both halves of what it claims.
    """
    text = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    listed = OUTSIDE_INDEX.search(text)
    assert listed, "AGENTS.md no longer names what the index leaves out, in the form this reads"
    names = [n.strip().strip("`") for n in listed.group(1).replace("\n", " ").split(",")]
    assert len(names) >= 5, names

    index_path = ROOT / "site" / "assets" / "api-index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        wrong = [n for n in names if n in index]
        assert not wrong, ("these are in the index after all, so the document is telling a "
                           f"reader to ignore it for nothing: {wrong}")

    binding = (ROOT / "borch_webgpu" / "__init__.py").read_text(encoding="utf-8")
    cpu = (ROOT / "borch_cpu.py").read_text(encoding="utf-8")
    absent = [n for n in names if n not in binding and n not in cpu]
    assert not absent, ("AGENTS.md names these as real and neither browser surface "
                        f"mentions them: {absent}")
