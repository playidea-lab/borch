"""Serves the explainer, the lessons, the API reference and the playground.

    npm run build:ts          # a browser cannot read TypeScript as it stands
    python3 site/build_api.py # pulls the API index out of the declaration files
    python3 site/serve.py     # opens http://127.0.0.1:8123/site/

**It serves the repository root**, because the pages ask for `../borch-ts/dist`. Serve
only `site/` and that path leaves the server, giving a 404 whose wording on screen is
indistinguishable from "there is no build".

The same arrangement as `borch-ts/test/run.py`; all that differs is that a person is
looking at this one rather than a runner.
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
        # **Caching off.** A fixed file hidden behind the browser's old copy is
        # indistinguishable from one that was never fixed — the search order was once
        # changed and then stared at as "nothing happened" for a long while. On a
        # development server that confusion costs far more than caching saves.
        self.send_header("Cache-Control", "no-store, must-revalidate")
        super().end_headers()

    def send_error(self, code, message=None, explain=None):
        # An unexplained 404 is not left covered up — a runner in this repository once
        # took 404 HTML as JavaScript and blew up somewhere else entirely.
        if code == 404:
            print(f"  [404] {self.path}", file=sys.stderr)
        super().send_error(code, message, explain)


def main():
    dist = ROOT / "borch-ts" / "dist" / "src" / "index.js"
    if not dist.exists():
        print("no build — the demo appears but Run does nothing.")
        print("  first: npm run build:ts")
    if not (ROOT / "site" / "assets" / "api.json").exists():
        print("no API index — the reference comes up empty.")
        print("  first: python3 site/build_api.py")
    if not (ROOT / "site" / "assets" / "data" / "cifar-train.jpg").exists():
        print("no tutorial data — 4 and 5 will not run (the rest will).")
        print("  first: python3 site/fetch_data.py [--download]")

    handler = functools.partial(Handler, directory=str(ROOT))
    socketserver.ThreadingTCPServer.allow_reuse_address = True
    with socketserver.ThreadingTCPServer(("127.0.0.1", PORT), handler) as httpd:
        url = f"http://127.0.0.1:{PORT}{PAGE}"
        print(f"open: {url}   (Ctrl+C to stop)")
        print(f"Korean: {url}ko/")
        webbrowser.open(url)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nstopping.")


if __name__ == "__main__":
    main()
