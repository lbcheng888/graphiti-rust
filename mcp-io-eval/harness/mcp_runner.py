from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, IO, List, Optional

from .launcher import RunningProcess


@dataclass
class RpcMessage:
    json: Dict[str, Any]
    raw: str
    ts: float


class JsonRpcClient:
    def __init__(self, stdin: IO[bytes], stdout: IO[bytes], log_path: Path):
        self._stdin = stdin
        self._stdout = stdout
        self._log_path = log_path
        self._id = 0
        self._lock = threading.Lock()
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._incoming: List[RpcMessage] = []
        self._reader.start()

    def _read_loop(self):
        with open(self._log_path, "w", encoding="utf-8") as logf:
            while True:
                line = self._stdout.readline()
                if not line:
                    break
                try:
                    s = line.decode("utf-8", errors="ignore").strip()
                except Exception:
                    s = ""
                if not s:
                    continue
                ts = time.time()
                try:
                    obj = json.loads(s)
                except Exception:
                    obj = {"_unparsed": s}
                self._incoming.append(RpcMessage(json=obj, raw=s, ts=ts))
                logf.write(s + "\n")
                logf.flush()

    def request(self, method: str, params: Dict[str, Any], timeout_sec: int = 30) -> Dict[str, Any]:
        with self._lock:
            self._id += 1
            rid = self._id
            payload = {"jsonrpc": "2.0", "id": rid, "method": method, "params": params}
            data = (json.dumps(payload) + "\n").encode("utf-8")
            # 发送前写入会话日志，形成原始事件流（含发送）
            try:
                with open(self._log_path, "a", encoding="utf-8") as logf:
                    logf.write(json.dumps({"_sent": payload, "ts": time.time()}, ensure_ascii=False) + "\n")
            except Exception:
                pass
            self._stdin.write(data)
            self._stdin.flush()
        # 简单等待匹配 id 的响应
        start = time.time()
        while time.time() - start < timeout_sec:
            for m in list(self._incoming):
                if m.json.get("id") == rid and ("result" in m.json or "error" in m.json):
                    return m.json
            time.sleep(0.01)
        raise TimeoutError(f"等待方法 {method} 响应超时")


def snapshot_tools(client: JsonRpcClient, list_method: str, out_path: Path, timeout_sec: int = 30) -> Dict[str, Any]:
    resp = client.request(list_method, params={}, timeout_sec=timeout_sec)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(resp, ensure_ascii=False, indent=2), encoding="utf-8")
    return resp


def run_case(client: JsonRpcClient, call_method: str, tool_name: str, arguments: Dict[str, Any], timeout_sec: int = 60) -> Dict[str, Any]:
    params = {"name": tool_name, "arguments": arguments}
    return client.request(call_method, params=params, timeout_sec=timeout_sec)


def open_client(rp: RunningProcess, session_log_path: Path) -> JsonRpcClient:
    assert rp.proc.stdin and rp.proc.stdout
    return JsonRpcClient(stdin=rp.proc.stdin, stdout=rp.proc.stdout, log_path=session_log_path)
