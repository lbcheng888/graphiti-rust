from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from harness.differ import write_case_report  # noqa: E402


def main() -> int:
    if len(sys.argv) < 5:
        print("用法: python3 scripts/diff_two.py <case_name> <left.json> <right.json> <out_dir>")
        return 2
    case = sys.argv[1]
    left = Path(sys.argv[2]).read_text(encoding="utf-8")
    right = Path(sys.argv[3]).read_text(encoding="utf-8")
    out_dir = Path(sys.argv[4])

    write_case_report(case, left, right, left, right, out_dir)
    print(f"已写入 {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
