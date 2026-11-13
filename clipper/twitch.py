"""Twitch Helix API: tokens, login, live status, VODs and stream markers.

App token (client credentials) is enough for /users and /videos (finding the VOD).
A user token (scope user:read:broadcast, from --twitch-login) is only needed for markers.
"""
from __future__ import annotations

import json
import secrets
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer

from .config import ENV_FILE, TWITCH_LOGIN_PORT, TWITCH_REDIRECT, env, save_env
from .util import ClipperError, log

_app_token: str | None = None


# ----------------------------------------------------------------------------
# Tokens
# ----------------------------------------------------------------------------
def _oauth(params: dict) -> dict:
    data = urllib.parse.urlencode(params).encode()
    req = urllib.request.Request("https://id.twitch.tv/oauth2/token", data=data)
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.load(r)


def client_creds() -> tuple[str, str]:
    cid, sec = env("TWITCH_CLIENT_ID"), env("TWITCH_CLIENT_SECRET")
    if not (cid and sec):
        raise ClipperError(f"TWITCH_CLIENT_ID and TWITCH_CLIENT_SECRET must be set in {ENV_FILE}")
    return cid, sec


def app_token(refresh: bool = False) -> str:
    """App token (no login needed) - enough to find your VODs."""
    global _app_token
    if refresh or not _app_token:
        cid, sec = client_creds()
        _app_token = _oauth({"client_id": cid, "client_secret": sec,
                             "grant_type": "client_credentials"})["access_token"]
    return _app_token


def user_token(refresh: bool = False) -> str:
    """Your own token (from --twitch-login) - needed to read stream markers."""
    if refresh or not env("TWITCH_TOKEN"):
        if not env("TWITCH_REFRESH_TOKEN"):
            raise ClipperError("no Twitch login yet - run:  python stream_clipper.py --twitch-login")
        cid, sec = client_creds()
        tok = _oauth({"client_id": cid, "client_secret": sec, "grant_type": "refresh_token",
                      "refresh_token": env("TWITCH_REFRESH_TOKEN")})
        save_env("TWITCH_TOKEN", tok["access_token"])
        save_env("TWITCH_REFRESH_TOKEN", tok.get("refresh_token") or env("TWITCH_REFRESH_TOKEN"))
    return env("TWITCH_TOKEN")


def helix(path: str, params: dict, token_fn=app_token) -> dict:
    """GET a Helix endpoint; on 401, retry once with a fresh token."""
    url = f"https://api.twitch.tv/helix/{path}?{urllib.parse.urlencode(params)}"
    for attempt in (1, 2):
        req = urllib.request.Request(url, headers={
            "Client-Id": client_creds()[0], "Authorization": f"Bearer {token_fn(refresh=attempt == 2)}"})
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code != 401 or attempt == 2:
                raise
    raise RuntimeError("unreachable")


# ----------------------------------------------------------------------------
# One-time login (authorization-code flow through a local server)
# ----------------------------------------------------------------------------
def twitch_login() -> None:
    """Browser login that stores TWITCH_TOKEN/TWITCH_REFRESH_TOKEN in .env."""
    cid, sec = client_creds()
    state = secrets.token_urlsafe(16)
    url = "https://id.twitch.tv/oauth2/authorize?" + urllib.parse.urlencode({
        "client_id": cid, "redirect_uri": TWITCH_REDIRECT, "response_type": "code",
        "scope": "user:read:broadcast", "state": state})
    result: dict = {}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            if "code" not in q and "error" not in q:
                self.send_response(404)
                self.end_headers()
                return
            if q.get("state", [""])[0] == state and "code" in q:
                result["code"] = q["code"][0]
                msg = "Logged in to stream_clipper - you can close this tab."
            else:
                result["error"] = q.get("error_description", ["login refused"])[0]
                msg = "Login failed: " + result["error"]
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write(msg.encode())

        def log_message(self, *a):
            pass

    server = HTTPServer(("localhost", TWITCH_LOGIN_PORT), Handler)
    log("opening Twitch in your browser...")
    log(f"if nothing opens, go to: {url}")
    webbrowser.open(url)
    while not result:
        server.handle_request()
    server.server_close()
    if "code" not in result:
        raise ClipperError(result["error"])
    tok = _oauth({"client_id": cid, "client_secret": sec, "code": result["code"],
                  "grant_type": "authorization_code", "redirect_uri": TWITCH_REDIRECT})
    save_env("TWITCH_TOKEN", tok["access_token"])
    save_env("TWITCH_REFRESH_TOKEN", tok["refresh_token"])
    log(f"logged in - tokens saved to {ENV_FILE}")


# ----------------------------------------------------------------------------
# Queries
# ----------------------------------------------------------------------------
def user_id(login: str) -> str:
    users = helix("users", {"login": login})["data"]
    if not users:
        raise ClipperError(f"Twitch channel '{login}' not found")
    return users[0]["id"]


def recent_archives(uid: str, first: int = 5) -> list[dict]:
    """Newest past broadcasts first. Each has id, url, stream_id, created_at..."""
    return helix("videos", {"user_id": uid, "type": "archive", "first": first})["data"]


def markers(vod_id: str | None) -> list[tuple[float, str]]:
    """(seconds into the VOD, description) for each stream marker. Never raises."""
    if not vod_id:
        log("--markers: no VOD id, skipping")
        return []
    try:
        data = helix("streams/markers", {"video_id": vod_id, "first": 100}, user_token)
    except Exception as e:  # noqa: BLE001
        log(f"--markers: skipped ({e})")
        return []
    marks = [(float(m["position_seconds"]), m.get("description") or "")
             for u in data.get("data", []) for v in u.get("videos", []) for m in v.get("markers", [])]
    log(f"markers: {len(marks)} found")
    return marks
