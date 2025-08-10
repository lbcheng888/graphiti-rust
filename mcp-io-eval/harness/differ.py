from __future__ import annotations

import difflib
import json
from pathlib import Path
from typing import Any, Dict


def json_diff(a: Any, b: Any) -> Dict[str, Any]:
    # 简单结构 diff：直接返回两侧，便于机器消费
    return {"left": a, "right": b}


def text_diff(a: str, b: str) -> str:
    al = a.splitlines(keepends=True)
    bl = b.splitlines(keepends=True)
    diff = difflib.unified_diff(al, bl, fromfile="left", tofile="right")
    return "".join(diff)


def write_case_report(case_name: str, left_raw: str, right_raw: str, left_canon: str, right_canon: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # 文本 diff
    txt = text_diff(left_canon, right_canon)
    (out_dir / f"{case_name}.diff.txt").write_text(txt, encoding="utf-8")

    # 结构 diff JSON
    try:
        a = json.loads(left_canon)
        b = json.loads(right_canon)
    except Exception:
        a = {"_raw": left_canon}
        b = {"_raw": right_canon}
    jd = json_diff(a, b)
    (out_dir / f"{case_name}.diff.json").write_text(json.dumps(jd, ensure_ascii=False, indent=2), encoding="utf-8")

    # 人可读 Markdown
    md = [f"# 用例 {case_name} 对比报告", "", "## 规范化差异 (unified)", "", "```diff", txt, "```", ""]
    (out_dir / f"{case_name}.report.md").write_text("\n".join(md), encoding="utf-8")
