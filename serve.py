#!/usr/bin/env python3
"""
serve.py — 本地开发服务器，带 COOP/COEP 头 (SharedArrayBuffer 依赖)
===================================================================
用法:
  python serve.py              # 默认 http://localhost:8090
  python serve.py --port 9000  # 自定义端口

同时代理 /decide 请求到 pipeline.py 的 FastAPI 后端 (localhost:8000)。
"""
import os, sys, argparse, functools, json
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.request import urlopen, Request
from urllib.error import URLError

PIPELINE_URL = "http://localhost:8000"


class COOPCOEPHandler(SimpleHTTPRequestHandler):
    """加 COOP/COEP + CORS 头的静态文件服务器，带 /decide 反代。"""

    def __init__(self, *args, directory=None, **kwargs):
        super().__init__(*args, directory=directory, **kwargs)

    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_POST(self):
        if self.path == "/decide":
            self._proxy_decide()
        else:
            self.send_error(404)

    def _proxy_decide(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length else b"{}"
        try:
            req = Request(
                f"{PIPELINE_URL}/decide",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(req, timeout=10) as resp:
                data = resp.read()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(data)
        except URLError as e:
            err = json.dumps({"error": str(e)}).encode()
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(err)


def main():
    parser = argparse.ArgumentParser(description="COOP/COEP dev server")
    parser.add_argument("--port", type=int, default=8090)
    args = parser.parse_args()

    # 切到项目根目录，这样 /brain-game/ 和 /shared/ 路径都能正确服务
    root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(root)

    handler = functools.partial(COOPCOEPHandler, directory=root)
    httpd = HTTPServer(("", args.port), handler)
    print(f"[serve] http://localhost:{args.port}/brain-game/game.html")
    print(f"[serve] COOP/COEP enabled · proxying /decide → {PIPELINE_URL}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[serve] stopped.")


if __name__ == "__main__":
    main()
