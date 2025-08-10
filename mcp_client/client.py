#!/usr/bin/env python3
import argparse
import json
import os
import shlex
import subprocess
import sys
from typing import Any, Dict

# 轻量 MCP 客户端：通过 stdio 启动指定的 MCP Server，基于 modelcontextprotocol 进行工具枚举与调用。
# 仅用于本地 CLI 环境，不进行任何联网行为。

try:
    from modelcontextprotocol import StdioServerParameters
    from modelcontextprotocol.client import MCPClient
except Exception as e:
    print("缺少依赖，请先运行 scripts/create_python_env.sh 安装 requirements.txt", file=sys.stderr)
    raise


def run_server_via_stdio(cmd: str) -> StdioServerParameters:
    # 将 shell 命令解析为 argv，避免使用 shell=True
    argv = shlex.split(cmd)
    return StdioServerParameters(command=argv[0], args=argv[1:])


def list_tools(server_cmd: str) -> None:
    params = run_server_via_stdio(server_cmd)
    with MCPClient(params) as client:
        tools = client.list_tools()
        print(json.dumps({"tools": tools}, ensure_ascii=False, indent=2))


def call_tool(server_cmd: str, name: str, args: str) -> None:
    params = run_server_via_stdio(server_cmd)
    try:
        payload: Dict[str, Any] = json.loads(args) if args else {}
    except json.JSONDecodeError as e:
        print(f"--args 不是合法 JSON: {e}", file=sys.stderr)
        sys.exit(2)
    with MCPClient(params) as client:
        result = client.call_tool(name=name, arguments=payload)
        print(json.dumps({"result": result}, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="轻量 MCP 客户端（stdio）")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list-tools", help="枚举 MCP 服务器工具")
    p_list.add_argument("--server-cmd", required=True, help="通过 stdio 启动的 MCP Server 命令")

    p_call = sub.add_parser("call-tool", help="调用指定工具")
    p_call.add_argument("--server-cmd", required=True, help="通过 stdio 启动的 MCP Server 命令")
    p_call.add_argument("--name", required=True, help="工具名")
    p_call.add_argument("--args", default="{}", help="JSON 字符串作为参数")

    args = parser.parse_args()

    if args.cmd == "list-tools":
        list_tools(args.server_cmd)
    elif args.cmd == "call-tool":
        call_tool(args.server_cmd, args.name, args.args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

