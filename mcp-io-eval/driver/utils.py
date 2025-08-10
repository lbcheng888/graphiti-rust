import json
import os
from datetime import datetime
from typing import Any, Dict, Iterable, List, Tuple, Optional

# 通用工具：快照写入、JSON 美化、路径确保


def ensure_dir_for(path: str):
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def pretty_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True)


Snapshot = Tuple[str, Dict[str, Any], Dict[str, Any]]


def dump_snapshots(path: str, snapshots: Iterable[Snapshot], extra: Optional[Dict[str, Any]] = None):
    ensure_dir_for(path)
    serializable: List[Dict[str, Any]] = []
    for label, req, resp in snapshots:
        serializable.append({
            "label": label,
            "request": req,
            "response": resp,
        })
    payload = {
        "version": 1,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "snapshots": serializable,
    }
    if extra:
        payload["extra"] = extra
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

