from __future__ import annotations

import json
import os
import socket
import time
from http.client import HTTPSConnection, IncompleteRead
from pathlib import Path
from socket import timeout as SocketTimeout
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPSHandler, ProxyHandler, Request, build_opener, urlopen

ROOT = Path(__file__).resolve().parents[1]

LLM_ENDPOINT = os.environ.get(
    "REFINX_LLM_ENDPOINT", "http://127.0.0.1:3800/v1/chat/completions",
)


def _pinned_hosts():


    pins = {}
    for item in os.environ.get("REFINX_PINNED_HOSTS", "").split(","):
        host, _, address = item.partition("=")
        if address:
            pins[host.strip()] = address.strip()
    return pins


class _PinnedConnection(HTTPSConnection):

    def connect(self):
        sock = socket.create_connection(
            (_pinned_hosts()[self.host], self.port), self.timeout
        )
        self.sock = self._context.wrap_socket(sock, server_hostname=self.host)


class _PinnedHandler(HTTPSHandler):
    def https_open(self, req):
        return self.do_open(_PinnedConnection, req, context=self._context)


_pinned_open = build_opener(ProxyHandler({}), _PinnedHandler()).open


def load_key(env_name, file_name):
    key = os.environ.get(env_name, "").strip()
    if key:
        return key

    key_dir = Path(os.environ.get("REFINX_KEY_DIR") or ROOT)
    with (key_dir / file_name).open(encoding="utf-8-sig") as handle:
        for line in handle:
            key = line.strip()
            if key and not key.startswith("```"):
                return key
    raise ValueError(f"No key configured for {env_name}")


def is_timeout(exc):
    return isinstance(exc, (SocketTimeout, TimeoutError)) or (
        isinstance(exc, URLError)
        and isinstance(exc.reason, (SocketTimeout, TimeoutError))
    )


def _request_json(request, timeout, open_url=None):
    for attempt in range(3):
        try:
            with (open_url or urlopen)(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            if not 500 <= exc.code < 600 or attempt == 2:
                raise
            exc.close()
        except (URLError, TimeoutError, ConnectionError, SocketTimeout, IncompleteRead) as exc:
            if is_timeout(exc) or attempt == 2:
                raise
        time.sleep((2, 5)[attempt])


def get_json(url, headers, timeout):
    pinned = urlsplit(url).hostname in _pinned_hosts()
    return _request_json(
        Request(url, headers=headers, method="GET"), timeout,
        _pinned_open if pinned else None,
    )


def post_json(url, headers, body, timeout):
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json", **headers}
    return _request_json(Request(url, data=data, headers=headers, method="POST"), timeout)
