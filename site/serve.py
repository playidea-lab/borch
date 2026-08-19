"""설명 페이지·강의·API 레퍼런스·플레이그라운드를 띄운다.

    npm run build:ts          # 브라우저는 TypeScript 를 그대로 못 읽는다
    python3 site/build_api.py # API 목록을 선언 파일에서 뽑는다
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

    def end_headers(self):
        # **캐시를 끈다.** 고친 파일이 브라우저의 옛 사본에 가려지면, 안 고쳐진 것과
        # 구별이 안 된다 — 실제로 검색 순서를 고쳐 놓고 "안 바뀌었다" 를 한참 봤다.
        # 개발 서버에서 캐시가 버는 시간보다 그 혼동이 훨씬 비싸다.
        self.send_header("Cache-Control", "no-store, must-revalidate")
        super().end_headers()

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
    if not (ROOT / "site" / "assets" / "api.json").exists():
        print("API 목록이 없다 — 레퍼런스가 빈 채로 뜬다.")
        print("  먼저: python3 site/build_api.py")
    if not (ROOT / "site" / "assets" / "data" / "cifar-train.jpg").exists():
        print("튜토리얼 데이터가 없다 — 4·5 번이 안 돈다(나머지는 돈다).")
        print("  먼저: python3 site/fetch_data.py [--download]")

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
