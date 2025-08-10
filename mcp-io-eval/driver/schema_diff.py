from typing import Any, Dict, List, Tuple, Optional, Set
import json
import os


# 仅对 inputSchema 的 subset 字段比对：
# type、required、properties 及其属性 type enum default description


KEEP_TOP_LEVEL = {"type", "required", "properties"}
KEEP_PROP_ATTRS = ["type", "enum", "default", "description"]


def normalize_schema(s: Any) -> Dict[str, Any]:
    if not isinstance(s, dict):
        return {}
    out: Dict[str, Any] = {k: s.get(k) for k in KEEP_TOP_LEVEL if k in s}
    if "properties" in out and isinstance(out["properties"], dict):
        props = {}
        for k, v in out["properties"].items():
            if isinstance(v, dict):
                props[k] = {kk: v.get(kk) for kk in KEEP_PROP_ATTRS if kk in v}
        out["properties"] = props
    if "required" in out and isinstance(out["required"], list):
        out["required"] = sorted(list({str(x) for x in out["required"]}))
    return out


def _diff_required(a: List[str], b: List[str]) -> Dict[str, List[str]]:
    sa, sb = set(a or []), set(b or [])
    return {
        "added_in_b": sorted(list(sb - sa)),
        "removed_in_b": sorted(list(sa - sb)),
        "equal": sa == sb,
    }


def _diff_properties(pa: Dict[str, Any], pb: Dict[str, Any]) -> Dict[str, Any]:
    keys_a, keys_b = set(pa.keys()), set(pb.keys())
    only_in_a = sorted(list(keys_a - keys_b))
    only_in_b = sorted(list(keys_b - keys_a))
    inter = keys_a & keys_b
    per_prop: Dict[str, Any] = {}
    for k in sorted(inter):
        va, vb = pa[k], pb[k]
        attr_diff = {}
        for attr in KEEP_PROP_ATTRS:
            if va.get(attr) != vb.get(attr):
                attr_diff[attr] = {"a": va.get(attr), "b": vb.get(attr)}
        per_prop[k] = {
            "equal": not bool(attr_diff),
            "attr_diff": attr_diff,
        }
    return {
        "only_in_a": only_in_a,
        "only_in_b": only_in_b,
        "per_property": per_prop,
    }


def diff_schema(a: Any, b: Any) -> Dict[str, Any]:
    na = normalize_schema(a)
    nb = normalize_schema(b)
    props_a = na.get("properties", {}) if isinstance(na.get("properties"), dict) else {}
    props_b = nb.get("properties", {}) if isinstance(nb.get("properties"), dict) else {}
    return {
        "equal": na == nb,
        "top_level_diff": {
            "type": {"a": na.get("type"), "b": nb.get("type"), "equal": na.get("type") == nb.get("type")},
            "required": _diff_required(na.get("required", []), nb.get("required", [])),
        },
        "properties_diff": _diff_properties(props_a, props_b),
        "normalized_a": na,
        "normalized_b": nb,
    }


def _load_mapping(path: Optional[str]) -> Dict[str, Any]:
    default_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cases.mapping.json")
    p = path or default_path
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def build_tools_index(tools: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    输入：name -> tool_meta，其中包含 inputSchema。
    输出：name -> inputSchema 的字典，若缺失则为空字典。
    """
    idx: Dict[str, Dict[str, Any]] = {}
    for name, meta in (tools or {}).items():
        schema = meta if isinstance(meta, dict) else {}
        # 允许直接传入 inputSchema 自身或包含 inputSchema 的对象
        if "inputSchema" in schema and isinstance(schema["inputSchema"], dict):
            schema = schema["inputSchema"]
        idx[name] = schema
    return idx


def diff_tools(a_tools: Dict[str, Any], b_tools: Dict[str, Any]) -> Dict[str, Any]:
    ia, ib = build_tools_index(a_tools), build_tools_index(b_tools)
    names_a, names_b = set(ia.keys()), set(ib.keys())
    only_in_a = sorted(list(names_a - names_b))
    only_in_b = sorted(list(names_b - names_a))
    inter = sorted(list(names_a & names_b))

    per_tool = {}
    for name in inter:
        per_tool[name] = diff_schema(ia[name], ib[name])

    return {
        "only_in_a": only_in_a,
        "only_in_b": only_in_b,
        "per_tool": per_tool,
    }


def render_markdown_report(
    a_tools: Dict[str, Any],
    b_tools: Dict[str, Any],
    *,
    mapping_path: Optional[str] = None,
    title: str = "Schema 对比报告",
) -> str:
    mapping = _load_mapping(mapping_path)
    sem_map = mapping.get("semantics_to_tools", {}) or {}

    # 基础集合差集
    d = diff_tools(a_tools, b_tools)
    lines: List[str] = []
    lines.append(f"# {title}")
    lines.append("")

    # 工具名集合差集
    lines.append("## 工具名集合差集")
    lines.append("")
    lines.append(f"- 仅在A中: {', '.join(d['only_in_a']) if d['only_in_a'] else '(无)'}")
    lines.append(f"- 仅在B中: {', '.join(d['only_in_b']) if d['only_in_b'] else '(无)'}")
    lines.append("")

    ia, ib = build_tools_index(a_tools), build_tools_index(b_tools)

    # 语义映射一致性矩阵
    lines.append("## 映射工具的一致性矩阵")
    lines.append("")
    lines.append("| 语义 | A 工具 | B 工具 | 状态 |")
    lines.append("|---|---|---|---|")
    for sem, pair in sem_map.items():
        a_name = pair.get("py") if "py" in pair else pair.get("a")
        b_name = pair.get("rs") if "rs" in pair else pair.get("b")
        a_schema = ia.get(a_name) if a_name else None
        b_schema = ib.get(b_name) if b_name else None
        if a_schema is None and b_schema is None:
            status = "两端均缺失"
        elif a_schema is None:
            status = "A 缺失"
        elif b_schema is None:
            status = "B 缺失"
        else:
            status = "相同" if normalize_schema(a_schema) == normalize_schema(b_schema) else "不同"
        lines.append(f"| {sem} | {a_name or '-'} | {b_name or '-'} | {status} |")
    lines.append("")

    # 每个属性的差异详情（仅对交集工具）
    lines.append("## 每个映射工具的属性差异详情")
    lines.append("")
    for sem, pair in sem_map.items():
        a_name = pair.get("py") if "py" in pair else pair.get("a")
        b_name = pair.get("rs") if "rs" in pair else pair.get("b")
        if not a_name or not b_name:
            continue
        if a_name not in ia or b_name not in ib:
            continue
        diff = diff_schema(ia[a_name], ib[b_name])
        lines.append(f"### {sem} ({a_name} ↔ {b_name})")
        lines.append("")
        # 顶层
        tdiff = diff["top_level_diff"]
        lines.append(f"- type: A={tdiff['type']['a']!r} B={tdiff['type']['b']!r} equal={tdiff['type']['equal']}")
        rq = tdiff["required"]
        lines.append(f"- required: equal={rq['equal']} added_in_b={rq['added_in_b']} removed_in_b={rq['removed_in_b']}")
        # 属性
        pd = diff["properties_diff"]
        lines.append(f"- properties 仅在A: {pd['only_in_a']}")
        lines.append(f"- properties 仅在B: {pd['only_in_b']}")
        if pd["per_property"]:
            lines.append("- 每个属性差异：")
            for pname, info in pd["per_property"].items():
                if info["equal"]:
                    continue
                lines.append(f"  - {pname}:")
                for attr, ab in info["attr_diff"].items():
                    lines.append(f"    - {attr}: A={ab['a']!r} B={ab['b']!r}")
        else:
            lines.append("- 每个属性差异：无")
        lines.append("")

    # 对交集但未在语义映射中的工具，也可列举基础差异
    other_inter = (set(ia.keys()) & set(ib.keys())) - {
        pair.get("py") if "py" in pair else pair.get("a") for pair in sem_map.values()
    } - {
        pair.get("rs") if "rs" in pair else pair.get("b") for pair in sem_map.values()
    }
    if other_inter:
        lines.append("## 非映射交集工具（简要差异）")
        lines.append("")
        for name in sorted(other_inter):
            diff = diff_schema(ia[name], ib[name])
            status = "相同" if diff["equal"] else "不同"
            lines.append(f"- {name}: {status}")
        lines.append("")

    return "\n".join(lines)
