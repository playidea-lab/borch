"""설명 페이지와 플레이그라운드를 띄운다.

    npm run build:ts          # 브라우저는 TypeScript 를 그대로 못 읽는다
    python3 site/serve.py     # http://127.0.0.1:8123/site/ 가 열린다

**저장소 루트를 얹는다** — 페이지가 `../borch-ts/dist` 를 부르기 때문이다. `site/`
만 얹으면 그 경로가 서버 밖으로 나가 404 가 되고, 화면에는 "방출물이 없다" 와
구별되지 않는 문구가 남는다.

`borch-ts/test/run.py` 와 같은 방식이고, 다른 점은 러너가 아니라 사람이 본다는 것뿐이다.
"""

import functools
import http.server
import pathlib
import socketserver
import sys
import webbrowser

ROOT = pathlib.Path(__file__).resolve().parents[1]
PORT = 8123
PAGE = "/site/"


class Handler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def send_error(self, code, message=None, explain=None):
        # 설명 안 된 404 를 덮어두지 않는다 — 이 저장소의 러너가 한 번 404 HTML 을
        # 자바스크립트로 받아 엉뚱한 자리에서 터진 적이 있다.
        if code == 404:
            print(f"  [404] {self.path}", file=sys.stderr)
        super().send_error(code, message, explain)


def main():
    dist = ROOT / "borch-ts" / "dist" / "src" / "index.js"
    if not dist.exists():
        print("방출물이 없다 — 데모는 뜨지만 Run 이 안 된다.")
        print("  먼저: npm run build:ts")

    handler = functools.partial(Handler, directory=str(ROOT))
    socketserver.ThreadingTCPServer.allow_reuse_address = True
    with socketserver.ThreadingTCPServer(("127.0.0.1", PORT), handler) as httpd:
        url = f"http://127.0.0.1:{PORT}{PAGE}"
        print(f"열렸다: {url}   (Ctrl+C 로 닫는다)")
        print(f"한국어: {url}ko/")
        webbrowser.open(url)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n닫는다.")


if __name__ == "__main__":
    main()
