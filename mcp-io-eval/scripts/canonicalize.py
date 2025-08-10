from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from harness.canonicalizer import canonicalize_json_text  # noqa: E402


def main() -> int:
    if len(sys.argv) < 3:
        print("用法: python3 scripts/canonicalize.py <in.json> <out.json>")
        return 2
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2])
    text = src.read_text(encoding="utf-8")
    canon = canonicalize_json_text(text)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(canon, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
