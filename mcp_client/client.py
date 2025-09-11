#!/usr/bin/env python3
import argparse
import json
import os
import shlex
import subprocess
import sys
from typing import Any, Dict, Optional

# 轻量 MCP 客户端：通过 stdio 启动指定的 MCP Server，基于 modelcontextprotocol 进行工具枚举与调用。
# 仅用于本地 CLI 环境，不进行任何联网行为。

HAS_MCP_LIB = True
try:
    from modelcontextprotocol import StdioServerParameters
    from modelcontextprotocol.client import MCPClient
except Exception:
    HAS_MCP_LIB = False


def _parse_server_cmd(cmd: str):
    argv = shlex.split(cmd)
    if "--stdio" not in argv:
        argv.append("--stdio")
    return argv


def run_server_via_stdio(cmd: str):
    # 统一的命令解析，若可用则返回 MCP 库参数，否则返回 argv 供降级实现使用
    argv = _parse_server_cmd(cmd)
    if HAS_MCP_LIB:
        return StdioServerParameters(command=argv[0], args=argv[1:])
    return argv


DEFAULT_TIMEOUT_SECS = int(os.getenv("MCP_CLIENT_TIMEOUT_SECS", "120"))


class _StdioRPC:
    """不依赖第三方库的最小化 MCP stdio 客户端实现。

    仅实现 initialize / initialized / tools/list / tools/call，足以做连通性与工具调用验证。
    """

    def __init__(self, argv, default_timeout: float = DEFAULT_TIMEOUT_SECS):
        self.argv = argv
        self.proc = None
        self._next_id = 1
        self.default_timeout = default_timeout

    def __enter__(self):
        self.proc = subprocess.Popen(
            self.argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=sys.stderr,
            bufsize=0,
        )
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            if self.proc and self.proc.stdin:
                self.proc.stdin.close()
        except Exception:
            pass
        try:
            if self.proc:
                self.proc.terminate()
        except Exception:
            pass

    @staticmethod
    def _write_frame(w, payload: bytes):
        header = f"Content-Length: {len(payload)}\r\n\r\n".encode()
        w.write(header)
        w.write(payload)
        w.flush()

    @staticmethod
    def _read_exact(r, n: int, timeout: float) -> bytes:
        import select
        buf = bytearray()
        remaining = n
        while remaining > 0:
            ready, _, _ = select.select([r], [], [], timeout)
            if not ready:
                raise TimeoutError("读取响应超时（body）")
            chunk = r.read(remaining)
            if not chunk:
                raise RuntimeError("子进程 stdout 提前结束")
            buf.extend(chunk)
            remaining -= len(chunk)
        return bytes(buf)

    @staticmethod
    def _read_headers(r, timeout: float) -> dict:
        import select
        headers = {}
        while True:
            ready, _, _ = select.select([r], [], [], timeout)
            if not ready:
                raise TimeoutError("读取响应超时（headers）")
            line = r.readline()
            if not line:
                raise RuntimeError("子进程 stdout 提前结束（headers）")
            # 标准 LSP 风格以 \r\n 结尾
            if line in (b"\r\n", b"\n"):
                break
            try:
                k, v = line.decode().split(":", 1)
                headers[k.strip().lower()] = v.strip()
            except Exception:
                # 非法头部：忽略
                continue
        return headers

    def _rpc(self, method: str, params=None, timeout: Optional[float] = None):
        if not self.proc or not self.proc.stdin or not self.proc.stdout:
            raise RuntimeError("进程未启动")
        if timeout is None:
            timeout = self.default_timeout
        rid = self._next_id
        self._next_id += 1
        req = {
            "jsonrpc": "2.0",
            "id": rid,
            "method": method,
        }
        if params is not None:
            req["params"] = params
        data = json.dumps(req, ensure_ascii=False).encode()
        self._write_frame(self.proc.stdin, data)
        # 读取响应
        headers = self._read_headers(self.proc.stdout, timeout)
        cl = headers.get("content-length")
        if not cl:
            raise RuntimeError("缺少 Content-Length 响应头")
        body = self._read_exact(self.proc.stdout, int(cl), timeout)
        try:
            return json.loads(body.decode())
        except Exception as e:
            raise RuntimeError(f"解析 JSON 响应失败: {e}; 原始: {body[:200]!r}")

    # 高层封装
    def initialize(self, timeout: Optional[float] = None):
        return self._rpc("initialize", {}, timeout)

    def initialized(self, timeout: Optional[float] = None):
        return self._rpc("initialized", {}, timeout)

    def tools_list(self, timeout: Optional[float] = None):
        return self._rpc("tools/list", {}, timeout)

    def tools_call(self, name: str, arguments: dict, timeout: Optional[float] = None):
        return self._rpc("tools/call", {"name": name, "arguments": arguments}, timeout)


def list_tools(server_cmd: str, timeout_secs: int = DEFAULT_TIMEOUT_SECS) -> None:
    params_or_argv = run_server_via_stdio(server_cmd)
    if HAS_MCP_LIB and isinstance(params_or_argv, StdioServerParameters):
        with MCPClient(params_or_argv) as client:
            tools = client.list_tools()
            print(json.dumps({"tools": tools}, ensure_ascii=False, indent=2))
    else:
        # 降级：直接通过 stdio RPC 调用
        with _StdioRPC(params_or_argv, default_timeout=timeout_secs) as rpc:
            rpc.initialize(timeout=timeout_secs)
            rpc.initialized(timeout=timeout_secs)
            resp = rpc.tools_list(timeout=timeout_secs)
            # 兼容不同字段形态
            tools = resp.get("result", {}).get("tools") if isinstance(resp, dict) else None
            print(json.dumps({"tools": tools}, ensure_ascii=False, indent=2))


def call_tool(server_cmd: str, name: str, args: str, timeout_secs: int = DEFAULT_TIMEOUT_SECS) -> None:
    params_or_argv = run_server_via_stdio(server_cmd)
    try:
        payload: Dict[str, Any] = json.loads(args) if args else {}
    except json.JSONDecodeError as e:
        print(f"--args 不是合法 JSON: {e}", file=sys.stderr)
        sys.exit(2)
    if HAS_MCP_LIB and isinstance(params_or_argv, StdioServerParameters):
        with MCPClient(params_or_argv) as client:
            result = client.call_tool(name=name, arguments=payload)
            print(json.dumps({"result": result}, ensure_ascii=False, indent=2))
    else:
        with _StdioRPC(params_or_argv, default_timeout=timeout_secs) as rpc:
            rpc.initialize(timeout=timeout_secs)
            rpc.initialized(timeout=timeout_secs)
            resp = rpc.tools_call(name, payload, timeout=timeout_secs)
            print(json.dumps({"result": resp.get("result")}, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="轻量 MCP 客户端（stdio）")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list-tools", help="枚举 MCP 服务器工具")
    p_list.add_argument("--server-cmd", required=True, help="通过 stdio 启动的 MCP Server 命令")
    p_list.add_argument("--timeout-secs", type=int, default=DEFAULT_TIMEOUT_SECS, help="请求超时（秒），默认从 MCP_CLIENT_TIMEOUT_SECS 或 120 取值")

    p_call = sub.add_parser("call-tool", help="调用指定工具")
    p_call.add_argument("--server-cmd", required=True, help="通过 stdio 启动的 MCP Server 命令")
    p_call.add_argument("--name", required=True, help="工具名")
    p_call.add_argument("--args", default="{}", help="JSON 字符串作为参数")
    p_call.add_argument("--timeout-secs", type=int, default=DEFAULT_TIMEOUT_SECS, help="请求超时（秒），默认从 MCP_CLIENT_TIMEOUT_SECS 或 120 取值")

    args = parser.parse_args()

    if args.cmd == "list-tools":
        list_tools(args.server_cmd, args.timeout_secs)
    elif args.cmd == "call-tool":
        call_tool(args.server_cmd, args.name, args.args, args.timeout_secs)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
