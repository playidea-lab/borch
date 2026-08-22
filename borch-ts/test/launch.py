"""There is one place that launches a browser: `tests/browser/launch.py`.

It is imported rather than rewritten here. Two runner trees (the sister library's and
borch.ts's) launching a browser with different arguments stops two numbers laid side by
side from sharing a scale, and this repository has already been wrong at that place once —
a headless measurement was read as the GPU's number.

The dependency direction is borch.ts → tests. The golden reads from there too, so it is
not a new direction.
"""

import importlib.util
import pathlib

_PATH = pathlib.Path(__file__).resolve().parents[2] / "tests" / "browser" / "launch.py"
_spec = importlib.util.spec_from_file_location("browser_launch", _PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

FLAGS = _mod.FLAGS
# **Only `browser` is exported.** The non-closing version over there (`_open`) was made
# private, and re-exporting it here would make that pointless — with two doors, only one
# gets fixed.
browser = _mod.browser
is_software = _mod.is_software
refuse_if_software = _mod.refuse_if_software
warn_if_software = _mod.warn_if_software
