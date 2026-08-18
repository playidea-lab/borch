"""브라우저를 띄우는 자리는 `tests/browser/launch.py` 하나다.

여기서 다시 쓰지 않고 그것을 들여온다. 두 러너 트리(자매용·borch.ts 용)가 갈린
인자로 브라우저를 띄우면 나란히 놓은 두 수가 같은 잣대가 아니게 되고, 이 저장소는
이미 그 자리에서 한 번 틀렸다 — 헤드리스로 잰 수를 GPU 의 수로 읽었다.

의존 방향은 borch.ts → tests 다. 골든도 그쪽에서 읽으므로 새로 생긴 방향이 아니다.
"""

import importlib.util
import pathlib

_PATH = pathlib.Path(__file__).resolve().parents[2] / "tests" / "browser" / "launch.py"
_spec = importlib.util.spec_from_file_location("browser_launch", _PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

FLAGS = _mod.FLAGS
# **`browser` 하나만 내보낸다.** 저쪽에서 안 닫는 판(`_open`)을 비공개로 돌렸고,
# 여기서 그것을 다시 열어 주면 그 조치가 무의미해진다 — 문이 둘이면 하나만 고쳐진다.
browser = _mod.browser
is_software = _mod.is_software
refuse_if_software = _mod.refuse_if_software
warn_if_software = _mod.warn_if_software
