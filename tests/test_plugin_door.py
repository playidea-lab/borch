"""The Claude Code plugin in this repository — `.claude-plugin/` and `skills/borch/SKILL.md`.

The skill is the agent page as a tool: its Python recipes have to run on the numpy core,
and every borch-ts name its TypeScript recipe uses has to exist in the API index, so a
renamed export cannot leave a recipe that agents copy verbatim pointing at nothing."""
import json
import pathlib
import re
import sys
import textwrap

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SKILL = ROOT / "skills" / "borch" / "SKILL.md"
FENCE = re.compile(r"```(\w+)\n(.*?)```", re.S)


def test_the_manifests_parse_and_agree_on_the_skill():
    plugin = json.loads((ROOT / ".claude-plugin" / "plugin.json").read_text())
    market = json.loads((ROOT / ".claude-plugin" / "marketplace.json").read_text())
    assert plugin["name"] == "borch"
    entry = next(p for p in market["plugins"] if p["name"] == "borch")
    assert entry["source"] == "./"
    for skill in entry["skills"]:
        assert (ROOT / skill / "SKILL.md").exists(), f"{skill} has no SKILL.md"


def test_the_skill_has_a_name_and_a_description_that_says_when():
    text = SKILL.read_text(encoding="utf-8")
    head = text.split("---")[1]
    assert re.search(r"^name: borch$", head, re.M)
    desc = re.search(r"^description: (.+)$", head, re.M).group(1)
    assert "Use when" in desc and "browser" in desc and "Pyodide" in desc, "the description is the trigger"


def test_the_python_recipes_in_the_skill_run_on_the_numpy_core():
    sys.path.insert(0, str(ROOT))
    ran = 0
    for lang, body in FENCE.findall(SKILL.read_text(encoding="utf-8")):
        if lang != "python" or body.lstrip().startswith("%pip") or re.search(r"^import borch_webgpu", body, re.M):
            continue  # the notebook cell and the ONNX export import the binding; the browser checks cover them
        exec(compile(textwrap.dedent(body), "SKILL.md", "exec"), {})  # noqa: S102 — the recipe as written
        ran += 1
    assert ran >= 1, "the training loop has to run on the core"


def test_the_onnx_recipe_names_the_binding_because_the_core_does_not_export():
    """`torch.onnx.export` exists in borch_webgpu and borch-ts, not in the numpy core — the
    first draft of this recipe imported `borch` and would have failed for every agent."""
    onnx = [b for lang, b in FENCE.findall(SKILL.read_text(encoding="utf-8")) if lang == "python" and "onnx.export" in b]
    assert len(onnx) == 1 and "import borch_webgpu as torch" in onnx[0]
    import borch  # noqa: PLC0415
    assert not hasattr(borch, "onnx"), "the core grew ONNX export — the recipe and AGENTS.md rule 10 can say so now"


def test_every_borch_ts_name_in_the_typescript_recipe_exists():
    index_path = ROOT / "site" / "assets" / "api-index.json"
    if not index_path.exists():
        pytest.skip("site/assets/api-index.json is built by site/build_api.py")
    index = json.loads(index_path.read_text())
    ts = next(b for lang, b in FENCE.findall(SKILL.read_text(encoding="utf-8")) if lang == "ts")
    imported = re.search(r"import \{ ([^}]+) \} from \"borch-ts\"", ts).group(1)
    names = [n.strip() for n in imported.split(",")]
    missing = [n for n in names if n not in index and n not in ("nn", "optim", "onnx")]
    assert not missing, f"imported from borch-ts but not in the API index: {missing}"
    for cls in re.findall(r"new (nn|optim)\.(\w+)", ts):
        assert index.get(cls[1], "").startswith(f"{cls[0]}."), f"{cls[0]}.{cls[1]} is not in the API index"
