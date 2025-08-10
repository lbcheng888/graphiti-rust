from __future__ import annotations

import json
import math
import re
from typing import Any, Dict, List, Tuple

VOLATILE_KEYS_DEFAULT = {"id", "timestamp", "ts", "request_id", "trace_id"}
FLOAT_PRECISION = 6


def _round_floats(obj: Any) -> Any:
    if isinstance(obj, float):
        if math.isfinite(obj):
            return round(obj, FLOAT_PRECISION)
        return obj
    if isinstance(obj, list):
        return [_round_floats(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _round_floats(v) for k, v in obj.items()}
    return obj


def _strip_volatile(obj: Any, volatile_keys: set[str]) -> Any:
    if isinstance(obj, dict):
        return {k: _strip_volatile(v, volatile_keys) for k, v in obj.items() if k not in volatile_keys}
    if isinstance(obj, list):
        return [_strip_volatile(x, volatile_keys) for x in obj]
    return obj


def _sort_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _sort_json(obj[k]) for k in sorted(obj.keys())}
    if isinstance(obj, list):
        return [_sort_json(x) for x in obj]
    return obj


def normalize_whitespace(s: str) -> str:
    # 折叠多空白为单空格，保留换行
    s = re.sub(r"[\t\x0b\x0c\r]+", " ", s)
    return s


def canonicalize_json_text(text: str, volatile_keys: Tuple[str, ...] = ()) -> str:
    try:
        data = json.loads(text)
    except Exception:
        # 尝试对每行 JSON 合并
        parts: List[Any] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                parts.append(json.loads(line))
            except Exception:
                parts.append({"_unparsed": normalize_whitespace(line)})
        data = parts
    data = _round_floats(data)
    data = _strip_volatile(data, set(VOLATILE_KEYS_DEFAULT).union(volatile_keys))
    data = _sort_json(data)
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def merge_stream_events(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    # 假定事件格式 {"event": "token"|..., "data": ...}
    merged: Dict[str, Any] = {"events": events}
    # 可扩展：根据事件类型合并重建最终文本
    texts = []
    for ev in events:
        if isinstance(ev, dict) and ev.get("event") == "token":
            v = ev.get("data")
            if isinstance(v, str):
                texts.append(v)
    if texts:
        merged["text"] = "".join(texts)
    return merged
