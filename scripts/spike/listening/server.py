#!/usr/bin/env python3
"""Loopback-only server for the blind listening test: static files, plus ratings appended to ratings.jsonl."""
import json, os, time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
HERE = os.path.dirname(os.path.abspath(__file__)); RATINGS = os.path.join(HERE, "ratings.jsonl")
class H(SimpleHTTPRequestHandler):
    def __init__(self, *a, **k): super().__init__(*a, directory=HERE, **k)
    def log_message(self, *a): pass
    def _json(self, code, obj):
        b = json.dumps(obj).encode(); self.send_response(code); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b))); self.send_header("Cache-Control", "no-store"); self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        if self.path.startswith("/api/ratings"):
            rows = [json.loads(l) for l in open(RATINGS)] if os.path.exists(RATINGS) else []
            return self._json(200, rows)
        return super().do_GET()
    def do_POST(self):
        if self.path != "/api/rate": return self._json(404, {})
        row = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        if not (isinstance(row.get("id"), str) and row.get("score") in (1, 2, 3, 4, 5)): return self._json(400, {"error": "id and score 1-5"})
        row["t"] = int(time.time())
        with open(RATINGS, "a") as f: f.write(json.dumps(row) + "\n")
        self._json(200, {"ok": True})
ThreadingHTTPServer(("127.0.0.1", 8790), H).serve_forever()
