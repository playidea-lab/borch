"""브라우저를 띄우는 자리 하나.

다섯 개의 러너가 각자 `p.chromium.launch(...)` 를 부르고 있었고 인자가 갈려 있었다 —
`tests/browser/run.py` 에는 플래그가 아예 없었다. macOS 에서는 그래도 WebGPU 가 나와서
안 보였는데, 갈린 조건으로 잰 두 수를 나란히 놓고 "같은 잣대" 라고 부르는 순간
그것은 거짓이 된다. 그래서 여기 모은다.

**`BORCH_CHROME_CHANNEL` 을 주면 시스템 브라우저를 쓴다.** Playwright 는 자기가 받은
Chromium 을 쓰는 것이 기본인데, GPU 서버에는 그것이 없고 배포판 Chrome 이 깔려 있는
경우가 있다. 브라우저를 하나 더 받게 하는 것보다 있는 것을 쓰는 편이 낫고, 무엇보다
**사용자가 실제로 쓰는 브라우저**가 그쪽이다.

    BORCH_CHROME_CHANNEL=chrome DISPLAY=:1 uv run ... --headed

`DISPLAY` 는 여기서 안 다룬다 — Playwright 가 프로세스 환경을 그대로 물려주므로 명령
앞에 붙이면 그만이다. 헤드리스로는 안 된다는 것이 이 저장소의 반복된 교훈이고
(소프트웨어 래스터라이저가 예외 없이 조용히 나온다), 러너들이 어댑터를 먼저 찍는
이유가 그것이다.
"""

import os
import re
import sys

# WebGPU 를 켜고 리눅스에서 Vulkan 을 쓰게 한다. macOS 는 Metal 이라 뒤엣것이 무시된다.
FLAGS = ["--enable-unsafe-webgpu", "--enable-features=Vulkan"]

# CPU 로 도는 구현들. Chrome 은 SwiftShader, 리눅스 Mesa 는 lavapipe(llvmpipe)다.
_SOFTWARE = re.compile(r"swiftshader|llvmpipe|lavapipe|software", re.I)


def is_software(adapter):
    """이 어댑터가 CPU 인가. 이름으로 가른다 — WebGPU 는 그것을 안 알려준다."""
    return bool(adapter and _SOFTWARE.search(str(adapter)))


def refuse_if_software(adapter, what):
    """**시간과 자원을 재는 것은 CPU 에서 무효다.** 무효면 True 를 준다.

    값은 장치가 안 바꾸므로 골든은 소프트웨어 어댑터에서도 유효한 증거다 — 로직이
    맞다는 증거이지 GPU 경로가 돈다는 증거가 아닐 뿐이다. 그래서 골든은 안 막고
    벤치·정확도만 막는다.

    이 구분이 이 함수가 있는 이유다. 리눅스 GPU 서버에서 헤드리스로 골든을 돌렸더니
    845/845 가 나왔는데 어댑터는 `google / swiftshader` 였다 — 통과는 진짜였지만
    "다른 벤더에서 확인했다"는 주장은 거짓이었다. 사람이 로그 첫 줄을 안 읽으면
    그대로 지나간다.
    """
    if not is_software(adapter):
        return False
    print(f"**소프트웨어 어댑터다({adapter}) — 이 장치에서 {what} 은 무효다.**\n"
          "  CPU 로 돈 것이라 이 수는 GPU 의 수가 아니다. `--headed` 로,\n"
          "  진짜 GPU 가 붙은 화면에서 다시 재라.", file=sys.stderr)
    return True


def warn_if_software(adapter, what):
    """골든처럼 **값만 묻는** 것에 쓴다. 막지 않고, 무엇을 증명했는지만 좁혀 적는다."""
    if is_software(adapter):
        print(f"  (소프트웨어 어댑터다 — {what} 이 맞다는 것은 증명되지만,\n"
              "   GPU 경로가 도는지는 이 실행이 증명하지 않는다.)")


def launch(playwright, headed=False, flags=FLAGS):
    """`headed` 가 아니면 소프트웨어 어댑터가 나온다 — 부르는 쪽이 어댑터를 찍어야 한다."""
    channel = os.environ.get("BORCH_CHROME_CHANNEL") or None
    return playwright.chromium.launch(
        headless=not headed,
        channel=channel,
        args=list(flags),
    )
