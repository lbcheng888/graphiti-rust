import json
import os
import subprocess
import threading
import time
from typing import Any, Dict, Optional, List, Tuple

CRLF = "\r\n"


class StdioMCPClient:
    """
    通过 stdio 与 MCP 服务器进行 JSON-RPC 通讯。
    - 读写使用 Content-Length 帧
    - 提供 initialize、tools/list、tools/call 的最小能力
    - 统一采集请求/响应快照，外部可用 utils.dump_snapshots 输出
    - 后台异步采集 stderr 方便排障
    """

    def __init__(self, cmd: List[str], cwd: Optional[str] = None, env: Optional[Dict[str, str]] = None, name: str = "server"):
        self.cmd = cmd
        self.cwd = cwd
        self.env = {**os.environ, **(env or {})}
        self.name = name
        self.proc: Optional[subprocess.Popen] = None
        self._id = 0
        self.snapshots: List[Tuple[str, Dict[str, Any], Dict[str, Any]]] = []
        self._stderr_thread: Optional[threading.Thread] = None
        self._stderr_lines: List[bytes] = []
        self._stderr_lock = threading.Lock()
        self._stopped = threading.Event()

    # -------------------------- process lifecycle ---------------------------
    def start(self):
        self.proc = subprocess.Popen(
            self.cmd,
            cwd=self.cwd,
            env=self.env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=False,
            bufsize=0,
        )
        # 后台采集 stderr
        def _pump_stderr():
            assert self.proc is not None
            while not self._stopped.is_set():
                chunk = self.proc.stderr.readline() if self.proc.stderr else b""
                if not chunk:
                    if self.proc.poll() is not None:
                        break
                    time.sleep(0.01)
                    continue
                with self._stderr_lock:
                    self._stderr_lines.append(chunk)
        self._stderr_thread = threading.Thread(target=_pump_stderr, name=f"{self.name}-stderr", daemon=True)
        self._stderr_thread.start()

    def stop(self):
        try:
            self._stopped.set()
            if self.proc and self.proc.poll() is None:
                self.proc.terminate()
        except Exception:
            pass

    # ------------------------------ io helpers ------------------------------
    def _next_id(self) -> int:
        self._id += 1
        return self._id

    def _write_msg(self, obj: Dict[str, Any]):
        if not self.proc or not self.proc.stdin:
            raise RuntimeError("process not started")
        body = json.dumps(obj, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        header = f"Content-Length: {len(body)}{CRLF}{CRLF}".encode("ascii")
        self.proc.stdin.write(header + body)
        self.proc.stdin.flush()

    def _read_msg(self) -> Dict[str, Any]:
        if not self.proc or not self.proc.stdout:
            raise RuntimeError("process not started")
        # 读取头
        header_bytes = b""
        while True:
            ch = self.proc.stdout.read(1)
            if not ch:
                raise RuntimeError("EOF from server")
            header_bytes += ch
            if header_bytes.endswith(b"\r\n\r\n"):
                break
        header_text = header_bytes.decode("ascii", errors="ignore")
        content_length = 0
        for line in header_text.split("\r\n"):
            if line.lower().startswith("content-length:"):
                try:
                    content_length = int(line.split(":", 1)[1].strip())
                except Exception:
                    content_length = 0
        # 读取 body
        body = b""
        remaining = content_length
        while remaining:
            chunk = self.proc.stdout.read(remaining)
            if not chunk:
                break
            body += chunk
            remaining -= len(chunk)
        if not body:
            return {}
        return json.loads(body.decode("utf-8"))

    # ------------------------------ RPC methods -----------------------------
    def initialize(self, protocol_version: str = "2024-11-05", capabilities: Optional[Dict[str, Any]] = None):
        req = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "initialize",
            "params": {
                "protocolVersion": protocol_version,
                "capabilities": capabilities or {"tools": {}},
            },
        }
        self._write_msg(req)
        resp = self._read_msg()
        self.snapshots.append(("initialize", req, resp))
        return resp

    def tools_list(self):
        req = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "tools/list",
            "params": {},
        }
        self._write_msg(req)
        resp = self._read_msg()
        self.snapshots.append(("tools/list", req, resp))
        return resp

    def tools_call(self, name: str, args: Optional[Dict[str, Any]] = None):
        req = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "tools/call",
            "params": {"name": name, "arguments": args or {}},
        }
        self._write_msg(req)
        resp = self._read_msg()
        self.snapshots.append((f"tools/call:{name}", req, resp))
        return resp

    # ------------------------------ diagnostics -----------------------------
    def get_stderr(self) -> str:
        with self._stderr_lock:
            return b"".join(self._stderr_lines).decode("utf-8", errors="replace")

