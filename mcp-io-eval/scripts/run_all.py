from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List
import time

try:
    import yaml  # type: ignore
except Exception:  # noqa: BLE001
    yaml = None

# 局部导入
sys.path.append(str(Path(__file__).resolve().parents[1]))
from harness.config import load_config  # noqa: E402
from harness.launcher import start_target, stop_target  # noqa: E402
from harness.mcp_runner import open_client, snapshot_tools, run_case  # noqa: E402
from harness.canonicalizer import canonicalize_json_text  # noqa: E402
from harness.differ import write_case_report  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
REPORTS = ROOT / "reports"
CASES = ROOT / "cases" / "cases.yaml"
TARGETS = ROOT / "harness" / "targets.yaml"


def load_cases(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    if path.suffix in (".yml", ".yaml"):
        if yaml is None:
            raise RuntimeError("未安装 PyYAML，无法解析 YAML。请执行: pip install pyyaml")
        obj = yaml.safe_load(text)
    else:
        obj = json.loads(text)
    return obj.get("cases", [])


def main() -> int:
    ROOT.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "raw").mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "canonical").mkdir(parents=True, exist_ok=True)
    (REPORTS / "logs").mkdir(parents=True, exist_ok=True)

    cfg = load_config(str(TARGETS))

    # 启动两个目标
    logs_dir = REPORTS / "logs"
    ra = start_target(cfg.target_a, logs_dir)
    rb = start_target(cfg.target_b, logs_dir)

    try:
        # 打开 MCP 客户端
        ca = open_client(ra, REPORTS / "logs" / f"{cfg.target_a.name}.session.log")
        cb = open_client(rb, REPORTS / "logs" / f"{cfg.target_b.name}.session.log")

        # 工具快照
        a_tools = snapshot_tools(ca, cfg.mcp_methods["list"], ARTIFACTS / "raw" / f"{cfg.target_a.name}.tools.json")
        b_tools = snapshot_tools(cb, cfg.mcp_methods["list"], ARTIFACTS / "raw" / f"{cfg.target_b.name}.tools.json")
        # 也输出规范化快照
        (ARTIFACTS / "canonical" / f"{cfg.target_a.name}.tools.json").write_text(
            canonicalize_json_text(json.dumps(a_tools, ensure_ascii=False)), encoding="utf-8"
        )
        (ARTIFACTS / "canonical" / f"{cfg.target_b.name}.tools.json").write_text(
            canonicalize_json_text(json.dumps(b_tools, ensure_ascii=False)), encoding="utf-8"
        )

        cases = load_cases(CASES)
        for case in cases:
            name = case["name"]
            tool = case["tool"]
            args = case.get("arguments", {})

            # 执行用例并计时
            t0 = time.perf_counter()
            a_resp = run_case(ca, cfg.mcp_methods["call"], tool, args)
            t1 = time.perf_counter()
            b_resp = run_case(cb, cfg.mcp_methods["call"], tool, args)
            t2 = time.perf_counter()

            a_elapsed_ms = int((t1 - t0) * 1000)
            b_elapsed_ms = int((t2 - t1) * 1000)

            # 保存原始
            (ARTIFACTS / "raw" / f"{name}.{cfg.target_a.name}.json").write_text(
                json.dumps(a_resp, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            (ARTIFACTS / "raw" / f"{name}.{cfg.target_b.name}.json").write_text(
                json.dumps(b_resp, ensure_ascii=False, indent=2), encoding="utf-8"
            )

            # 规范化
            a_canon = canonicalize_json_text(json.dumps(a_resp, ensure_ascii=False))
            b_canon = canonicalize_json_text(json.dumps(b_resp, ensure_ascii=False))
            (ARTIFACTS / "canonical" / f"{name}.{cfg.target_a.name}.json").write_text(a_canon, encoding="utf-8")
            (ARTIFACTS / "canonical" / f"{name}.{cfg.target_b.name}.json").write_text(b_canon, encoding="utf-8")

            # 差异
            write_case_report(
                case_name=name,
                left_raw=json.dumps(a_resp, ensure_ascii=False, indent=2),
                right_raw=json.dumps(b_resp, ensure_ascii=False, indent=2),
                left_canon=a_canon,
                right_canon=b_canon,
                out_dir=REPORTS / "cases",
            )

            # 记录用例元信息（含耗时与日志引用路径）
            case_meta = {
                "name": name,
                "tool": tool,
                "arguments": args,
                "timings_ms": {
                    cfg.target_a.name: a_elapsed_ms,
                    cfg.target_b.name: b_elapsed_ms,
                },
                "artifacts": {
                    "raw": {
                        cfg.target_a.name: str((ARTIFACTS / "raw" / f"{name}.{cfg.target_a.name}.json").relative_to(ROOT)),
                        cfg.target_b.name: str((ARTIFACTS / "raw" / f"{name}.{cfg.target_b.name}.json").relative_to(ROOT)),
                    },
                    "canonical": {
                        cfg.target_a.name: str((ARTIFACTS / "canonical" / f"{name}.{cfg.target_a.name}.json").relative_to(ROOT)),
                        cfg.target_b.name: str((ARTIFACTS / "canonical" / f"{name}.{cfg.target_b.name}.json").relative_to(ROOT)),
                    },
                    "logs": {
                        cfg.target_a.name: str((REPORTS / "logs" / f"{cfg.target_a.name}.session.log").relative_to(ROOT)),
                        cfg.target_b.name: str((REPORTS / "logs" / f"{cfg.target_b.name}.session.log").relative_to(ROOT)),
                        f"{cfg.target_a.name}.stderr": str((REPORTS / "logs" / f"{cfg.target_a.name}.stderr.log").relative_to(ROOT)),
                        f"{cfg.target_b.name}.stderr": str((REPORTS / "logs" / f"{cfg.target_b.name}.stderr.log").relative_to(ROOT)),
                    },
                    "diff": str((REPORTS / "cases" / f"{name}.diff.txt").relative_to(ROOT)),
                },
                "equal": a_canon == b_canon,
            }
            (REPORTS / "cases" / f"{name}.meta.json").write_text(
                json.dumps(case_meta, ensure_ascii=False, indent=2), encoding="utf-8"
            )

        # 汇总报告（简单列目录）
        summary = {
            "cases": [c["name"] for c in cases],
            "targets": [cfg.target_a.name, cfg.target_b.name],
            "tools_snapshots": {
                cfg.target_a.name: str((ARTIFACTS / "raw" / f"{cfg.target_a.name}.tools.json").relative_to(ROOT)),
                cfg.target_b.name: str((ARTIFACTS / "raw" / f"{cfg.target_b.name}.tools.json").relative_to(ROOT)),
            },
            "logs": {
                cfg.target_a.name: {
                    "session": str((REPORTS / "logs" / f"{cfg.target_a.name}.session.log").relative_to(ROOT)),
                    "stderr": str((REPORTS / "logs" / f"{cfg.target_a.name}.stderr.log").relative_to(ROOT)),
                },
                cfg.target_b.name: {
                    "session": str((REPORTS / "logs" / f"{cfg.target_b.name}.session.log").relative_to(ROOT)),
                    "stderr": str((REPORTS / "logs" / f"{cfg.target_b.name}.stderr.log").relative_to(ROOT)),
                },
            },
        }
        (REPORTS / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        (REPORTS / "README.md").write_text(
            "\n".join([
                "# 报告说明",
                "- logs/: 目标进程 stderr 与会话日志",
                "- cases/: 逐用例差异报告",
                "- summary.json: 汇总信息",
            ]),
            encoding="utf-8",
        )

        print("完成：请查看 artifacts/ 与 reports/ 目录。")
        return 0
    finally:
        # 关闭子进程
        code_a, _ = stop_target(ra, timeout_sec=10)
        code_b, _ = stop_target(rb, timeout_sec=10)
        print(f"targets exited: {cfg.target_a.name}={code_a}, {cfg.target_b.name}={code_b}")


if __name__ == "__main__":
    raise SystemExit(main())
