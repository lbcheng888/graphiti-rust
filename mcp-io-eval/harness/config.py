from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

try:
    import yaml  # type: ignore
except Exception:  # noqa: BLE001
    yaml = None  # 允许纯 JSON 环境


@dataclass
class TargetConfig:
    name: str
    cmd: List[str]
    cwd: Optional[str] = None
    env: Dict[str, str] = field(default_factory=dict)
    timeout_sec: int = 120
    needs_port: bool = False
    port_env_key: str = "PORT"
    ready_regex: Optional[str] = None  # 可选：判定服务已就绪的日志正则


@dataclass
class EvalConfig:
    target_a: TargetConfig
    target_b: TargetConfig
    mcp_methods: Dict[str, str] = field(default_factory=lambda: {
        "list": "tools/list",
        "call": "tools/call",
    })


def _as_dict(data: Any) -> Dict[str, Any]:
    if isinstance(data, dict):
        return data
    raise ValueError("配置文件应为字典对象")


def load_config(path: str) -> EvalConfig:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    obj: Dict[str, Any]
    if path.endswith(".json"):
        obj = json.loads(text)
    elif path.endswith(".yml") or path.endswith(".yaml"):
        if yaml is None:
            raise RuntimeError("未安装 PyYAML，无法解析 YAML。请执行: pip install pyyaml")
        obj = _as_dict(yaml.safe_load(text))
    else:
        # 优先尝试 YAML，再尝试 JSON
        if yaml is not None:
            try:
                obj = _as_dict(yaml.safe_load(text))
            except Exception:
                obj = json.loads(text)
        else:
            obj = json.loads(text)

    def parse_target(prefix: str) -> TargetConfig:
        t = _as_dict(obj[prefix])
        return TargetConfig(
            name=t.get("name", prefix),
            cmd=t["cmd"],
            cwd=t.get("cwd"),
            env=t.get("env", {}),
            timeout_sec=int(t.get("timeout_sec", 120)),
            needs_port=bool(t.get("needs_port", False)),
            port_env_key=t.get("port_env_key", "PORT"),
            ready_regex=t.get("ready_regex"),
        )

    mcp_methods = obj.get("mcp_methods", {"list": "tools/list", "call": "tools/call"})

    return EvalConfig(
        target_a=parse_target("target_a"),
        target_b=parse_target("target_b"),
        mcp_methods=mcp_methods,
    )
