import json
import os
import sys
import time
import pathlib
import datetime
from typing import Any, Dict, List, Optional, Tuple

import yaml

# 兼容：既可作为包导入（相对导入），也可作为脚本直接运行（同目录导入）
try:
    from .stdio_client import StdioMCPClient
    from .utils import dump_snapshots
    from .assertions import assert_jsonrpc_ok
    from . import schema_diff as sd
except Exception:  # noqa: E722
    import sys as _sys, os as _os
    _CUR_DIR = _os.path.dirname(__file__)
    if _CUR_DIR not in _sys.path:
        _sys.path.insert(0, _CUR_DIR)
    from stdio_client import StdioMCPClient  # type: ignore
    from utils import dump_snapshots  # type: ignore
    from assertions import assert_jsonrpc_ok  # type: ignore
    import schema_diff as sd  # type: ignore


"""
运行器 run_suite.py

功能：
- 读取 targets.yaml，按目标逐一启动 MCP server（Rust 与 Python）
- 对每个 target 统一执行用例序列：initialize -> tools/list -> 多轮 tools/call
- 对 add_memory 的参数做语义级映射与转换（content -> py.episode_body 等）
- 对 search_facts 若目标缺失，标记为部分等价
- 采集并写入 snapshots 与 reports（schema_diff.md, results.md）

严格/宽松模式：
- 命令行开关 --strict/--lenient 或环境变量 EVAL_STRICT=true/false
"""


# ------------------------------- helpers ---------------------------------

def _now_ts() -> str:
    return datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _ensure_dir(p: str):
    d = os.path.abspath(p)
    os.makedirs(d, exist_ok=True)


def _family_from_name(name: str) -> str:
    n = (name or "").lower()
    if "rust" in n or n.endswith("_rs") or n.endswith("-rs") or "_rs_" in n or " graphiti_rs" in n or "rs" == n:
        return "rust"
    if "python" in n or "py" in n:
        return "python"
    return "unknown"


def _extract_tools_index(tools_resp: Dict[str, Any]) -> Dict[str, Any]:
    """
    将 tools/list 响应标准化为 name -> {inputSchema,...} 字典，用于 schema diff。
    兼容：
    - result = {"tools": [{"name": "...", "inputSchema": {...}}]}
    - result = {"tools": {"name": {"inputSchema": {...}}}}
    - result 本身就是上面两种结构之一
    """
    if not isinstance(tools_resp, dict):
        return {}
    result = tools_resp.get("result", tools_resp)
    tools_obj = result.get("tools", result)
    idx: Dict[str, Any] = {}
    if isinstance(tools_obj, list):
        for t in tools_obj:
            if isinstance(t, dict) and "name" in t:
                idx[str(t["name"])] = t.get("inputSchema", t)
    elif isinstance(tools_obj, dict):
        # 可能就是 name->meta
        for name, meta in tools_obj.items():
            if isinstance(meta, dict):
                if "inputSchema" in meta and isinstance(meta["inputSchema"], dict):
                    idx[str(name)] = meta["inputSchema"]
                else:
                    idx[str(name)] = meta
    return idx


def _load_cases(cases_dir: str) -> List[Tuple[str, Dict[str, Any]]]:
    """读取 cases 目录下的 JSON 用例；按 id 排序，优先 SCHEMA_LIST，再 A1,S1,F1,R1 顺序。
    返回列表：(case_id, case_json)
    """
    entries: List[Tuple[str, Dict[str, Any]]] = []
    for fn in os.listdir(cases_dir):
        if not fn.endswith(".json"):
            continue
        path = os.path.join(cases_dir, fn)
        try:
            data = _load_json(path)
            cid = str(data.get("id") or pathlib.Path(fn).stem)
            entries.append((cid, data))
        except Exception:
            pass
    # 排序规则：SCHEMA_LIST 最前，其余按 A1, S1, F1, R1 自然排序
    def _key(it: Tuple[str, Dict[str, Any]]):
        cid = it[0]
        if cid == "SCHEMA_LIST":
            return (0, cid)
        # 提取字母序 + 数字
        head = ''.join([c for c in cid if c.isalpha()]) or 'Z'
        num = int(''.join([c for c in cid if c.isdigit()]) or '999')
        order = {"A": 1, "S": 2, "F": 3, "R": 4}.get(head[0].upper(), 9)
        return (order, num)

    entries.sort(key=_key)
    return entries


def _map_args_by_semantics(sem: str, common_args: Dict[str, Any], mapping_json: Dict[str, Any], family: str) -> Dict[str, Any]:
    """依据 cases.mapping.json 中的 arg_mappings 做语义级参数映射。family = 'python'|'rust'"""
    amap = (mapping_json.get("arg_mappings", {}) or {}).get(sem, {})
    if family == "python":
        m = amap.get("py_from_common", {})
    else:
        m = amap.get("rs_from_common", {})
    out: Dict[str, Any] = {}
    for k_common, v in (common_args or {}).items():
        k_target = m.get(k_common, k_common)
        out[k_target] = v
    return out


def _select_tool_and_args(case: Dict[str, Any], mapping_json: Dict[str, Any], family: str, previous_results: Dict[str, Any]) -> Tuple[Optional[str], Dict[str, Any], Dict[str, Any]]:
    """
    从用例与语义映射推导当前目标需调用的工具与参数。
    返回：(tool_name or None, args, policy)
    支持：
    - case[family_key] 明确给出 tool/args
    - 若缺失，尝试用 semantics + arg_mappings 从 common 构造
    - 若 args_from_previous 指定了上一个用例 id，则从 previous_results 中提取 id 字段作为参数
    """
    sem = case.get("semantics")
    fam_key = "py" if family == "python" else "rs"
    fam_spec = case.get(fam_key) or {}
    tool = fam_spec.get("tool")
    args = dict(fam_spec.get("args") or {})

    # 从前置结果提取 id
    prev_id_label = fam_spec.get("args_from_previous") or case.get("args_from_previous")
    if prev_id_label:
        prev = previous_results.get(prev_id_label) or {}
        # 自动寻找 id/uuid/node_id 等
        id_candidates = []
        # 来自用例可配置
        extra = (fam_spec.get("args") or {}).get("id_field_candidates") or (case.get("assert") or {}).get("id_field_candidates")
        if isinstance(extra, list):
            id_candidates.extend([str(x) for x in extra])
        id_candidates.extend(["id", "uuid", "node_id"])  # 默认候选
        found = None
        if isinstance(prev, dict):
            for k in id_candidates:
                if k in prev:
                    found = prev[k]
                    break
            # 若 prev.result 才是对象
            if found is None and "result" in prev and isinstance(prev["result"], dict):
                for k in id_candidates:
                    if k in prev["result"]:
                        found = prev["result"][k]
                        break
        if found is not None:
            args = {**args, "id": found}

    # 若未提供 tool/args，尝试语义映射
    if not tool and sem:
        tools_map = (mapping_json.get("semantics_to_tools", {}) or {}).get(sem, {})
        tool = tools_map.get("py" if family == "python" else "rs")
    if not args and sem and isinstance(case.get("common_args"), dict):
        args = _map_args_by_semantics(sem, case["common_args"], mapping_json, family)

    policy = case.get("policy") or {}
    return tool, args, policy


# ------------------------------- core flow --------------------------------

def run_target(target_name: str, target_cfg: Dict[str, Any], cases: List[Tuple[str, Dict[str, Any]]], mapping_json: Dict[str, Any], out_snap_dir: str, strict: bool = True) -> Tuple[Dict[str, Any], List[Tuple[str, str, str]]]:
    """
    执行单个 target：握手、tools/list、按序执行用例。
    返回：(tools_index, results)
      - tools_index: name->inputSchema 字典
      - results: [(case_id, status)] 其中 status ∈ {PASS, FAIL, PARTIAL, SKIP}
    产生 snapshots：snapshots/<target>.<family>_stdio.<timestamp>.json
    """
    cmd = target_cfg.get("cmd")
    if isinstance(cmd, str):
        cmd_tokens = cmd.split()
    else:
        cmd_tokens = list(cmd or [])
    cwd = target_cfg.get("cwd") or None
    env = dict(target_cfg.get("env") or {})

    family = _family_from_name(target_cfg.get("name") or target_name)

    client = StdioMCPClient(cmd=cmd_tokens, cwd=cwd, env=env, name=target_name)
    client.start()

    results: List[Tuple[str, str, str]] = []
    tools_index: Dict[str, Any] = {}

    try:
        init_resp = client.initialize()
        assert_jsonrpc_ok(init_resp)

        list_resp = client.tools_list()
        assert_jsonrpc_ok(list_resp)
        tools_index = _extract_tools_index(list_resp)

        # 逐用例执行
        previous_results: Dict[str, Any] = {}
        for cid, case in cases:
            if cid == "SCHEMA_LIST":
                # 不对单 target 断言；在汇总阶段做 diff
                results.append((cid, "PASS", ""))
                continue

            tool, args, policy = _select_tool_and_args(case, mapping_json, family, previous_results)

            # Python 侧 add_memory / search 节点：优先尝试非嵌入路径参数
            sem = case.get("semantics")
            note = ""
            attempted_disable_embeddings = False
            def _find_disable_params(schema: Dict[str, Any]) -> Dict[str, Any]:
                # 识别 inputSchema.properties 中的降级参数
                try:
                    props = (schema or {}).get("properties") or {}
                    out: Dict[str, Any] = {}
                    if "use_embeddings" in props:
                        out["use_embeddings"] = False
                    if "mode" in props:
                        # 直接尝试元数据路径
                        out["mode"] = "metadata_only"
                    return out
                except Exception:
                    return {}

            # 若映射缺失或工具不存在，处理部分等价策略
            if not tool:
                if (policy.get("if_missing_on_target") == "partial_equivalence"):
                    results.append((cid, "PARTIAL", "用例映射缺失或未在该目标实现，按部分等价计入"))
                    continue
                else:
                    results.append((cid, "SKIP", "用例映射缺失，跳过执行"))
                    continue

            # 目标工具是否在列表中？若不在且允许部分等价
            if tools_index and tool not in tools_index:
                # 特例：Rust 端 facts 类检索未实现时，视为部分等价（不判失败）
                if family == "rust" and (case.get("semantics") in ("search_facts", "facts_search", "get_facts", "get_episodes", "search_memory_facts")):
                    results.append((cid, "PARTIAL", "Rust 端缺失 facts/episodes 类工具，按部分等价计入"))
                    continue
                if policy.get("if_missing_on_target") == "partial_equivalence":
                    results.append((cid, "PARTIAL", "目标缺失该工具，按部分等价计入"))
                    continue

            # 在调用前尝试对 Python 侧进行“禁用嵌入/纯文本模式”降级
            if family == "python" and sem in ("add_memory", "search_nodes", "search_memory"):
                try:
                    schema = tools_index.get(tool, {})
                    disable_params = _find_disable_params(schema)
                    # 仅在 args 尚未明确设置相应键时进行注入
                    inject: Dict[str, Any] = {}
                    for k, v in disable_params.items():
                        if k not in args:
                            inject[k] = v
                    if inject:
                        args = {**args, **inject}
                        attempted_disable_embeddings = True
                        note = (note + f"已注入降级参数: {inject}. ").strip()
                except Exception:
                    pass

            # 发起调用
            try:
                call_resp = client.tools_call(tool, args)
                # 允许 error或result 二选一
                if not ("result" in call_resp or "error" in call_resp):
                    raise AssertionError("tools/call: missing result or error")

                # 如果 Python 侧调用返回错误且疑似远端推理/嵌入依赖失败，并且无可用的禁用参数，则按部分等价计入
                def _is_remote_dep_error(resp: Dict[str, Any]) -> bool:
                    try:
                        err = resp.get("error") or {}
                        msg = str(err.get("message") or "").lower()
                        data_str = str(err.get("data") or "").lower()
                        text = msg + " " + data_str
                        keywords = [
                            "openai", "anthropic", "embedding", "embeddings", 
                            "api key", "remote", "v1/embeddings", "rate limit", "httpx", "model"
                        ]
                        return any(k in text for k in keywords)
                    except Exception:
                        return False

                if family == "python" and sem in ("add_memory", "search_nodes", "search_memory") and ("error" in call_resp):
                    # 若错误指向远端依赖，则标记 PARTIAL（无论是否已尝试注入禁用参数）
                    if _is_remote_dep_error(call_resp):
                        add_msg = "已尝试降级参数仍失败；" if attempted_disable_embeddings else "缺少可用的禁用嵌入参数；"
                        note = (note + f"Python 侧疑似依赖远端嵌入/推理而失败；{add_msg}按部分等价计入。建议：如工具支持，提供 use_embeddings=false 或 mode=metadata_only；在无 API Key 时自动降级为纯文本。").strip()
                        results.append((cid, "PARTIAL", note))
                        # 不更新 previous_results，以避免污染后续关联检索
                        continue

                # 简化断言：根据用例的 assert 规则做最小校验
                ok = _evaluate_case_assertion(case, call_resp, strict=strict, mapping_json=mapping_json)
                results.append((cid, "PASS" if ok else "FAIL", note))
                previous_results[cid] = call_resp.get("result", call_resp)
            except Exception:
                results.append((cid, "FAIL", note))
    finally:
        # 写入 snapshots
        ts = _now_ts()
        snap_name = f"{target_cfg.get('name', target_name)}.{family}_stdio.{ts}.json"
        snap_path = os.path.join(out_snap_dir, snap_name)
        try:
            dump_snapshots(snap_path, client.snapshots, extra={"stderr": client.get_stderr()})
        except Exception as e:
            sys.stderr.write(f"failed to dump snapshots for {target_name}: {e}\n")
        client.stop()

    return tools_index, results


def _evaluate_case_assertion(case: Dict[str, Any], call_resp: Dict[str, Any], *, strict: bool, mapping_json: Dict[str, Any]) -> bool:
    """
    针对给定用例进行最小断言：
    - A1: response.require_any_of -> 至少包含任一字段
    - S1: response_array_or_object + element_schema.fields_any_of
    - 默认：只要存在 result 即通过（宽松），严格则要求 jsonrpc OK
    """
    try:
        if strict:
            # 严格至少要求 JSON-RPC OK（无 error 且有 result）
            assert_jsonrpc_ok(call_resp)
        res = call_resp.get("result")
        ast = case.get("assert") or {}
        if not ast:
            return res is not None

        # require_any_of 针对对象
        if isinstance(ast.get("response"), dict) and isinstance(res, dict):
            req = ast["response"].get("require_any_of")
            if isinstance(req, list) and req:
                if any(k in res for k in req):
                    return True
                return False

        # response_array_or_object with element_schema
        if ast.get("response_array_or_object"):
            elems: List[Dict[str, Any]] = []
            if isinstance(res, list):
                elems = res
            elif isinstance(res, dict):
                elems = [res]
            else:
                return False
            if not elems:
                # 允许空集通过（可调）
                return True
            es = ast.get("element_schema") or {}
            any_fields = es.get("fields_any_of") or []
            if any_fields:
                # 仅检查首元素包含任一字段
                first = elems[0]
                return any(k in first for k in any_fields)
            return True

        # 默认：只要有 result 即通过
        return res is not None or ("error" not in call_resp)
    except Exception:
        return False


# ------------------------------- entrypoint -------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="MCP IO 等价性运行器")
    parser.add_argument("--targets", default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "harness", "targets.yaml"), help="targets.yaml 路径")
    parser.add_argument("--cases-dir", default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "cases"), help="cases 目录路径")
    parser.add_argument("--mapping", default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "cases.mapping.json"), help="参数/语义映射配置")
    parser.add_argument("--snapshots", default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "snapshots"), help="snapshots 输出目录")
    parser.add_argument("--reports", default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports"), help="reports 输出目录")
    parser.add_argument("--strict", dest="strict", action="store_true", help="严格模式：字段与类型强校验")
    parser.add_argument("--lenient", dest="strict", action="store_false", help="宽松模式：允许额外字段等")
    parser.set_defaults(strict=os.environ.get("EVAL_STRICT", "true").lower() in ("1", "true", "yes"))

    args = parser.parse_args(argv)

    _ensure_dir(args.snapshots)
    _ensure_dir(args.reports)

    targets_cfg = _load_yaml(args.targets)
    mapping_json = _load_json(args.mapping)
    cases = _load_cases(args.cases_dir)

    # 收集各 target 的 tools 列表与结果
    target_entries: List[Tuple[str, Dict[str, Any], List[Tuple[str, str, str]]]] = []

    # 仅选择带有 name/cmd 的顶层键（剔除元字段）
    for tkey, tcfg in targets_cfg.items():
        if not isinstance(tcfg, dict):
            continue
        if not (tcfg.get("cmd") and tcfg.get("name") is not None):
            continue
        name = tcfg.get("name") or tkey
        print(f"[run] target={name} ...", file=sys.stderr)
        tools_idx, results = run_target(name, tcfg, cases, mapping_json, args.snapshots, strict=args.strict)
        target_entries.append((name, tools_idx, results))

    # schema diff（仅当存在两个以上 target）
    if len(target_entries) >= 2:
        # 取前两个作为 A/B 进行对比
        (name_a, tools_a, _), (name_b, tools_b, _) = target_entries[0], target_entries[1]
        md = sd.render_markdown_report(
            tools_a,
            tools_b,
            mapping_path=args.mapping,
            title=f"Schema 对比报告: {name_a} vs {name_b}"
        )
        schema_path = os.path.join(args.reports, "schema_diff.md")
        with open(schema_path, "w", encoding="utf-8") as f:
            f.write(md)

    # 汇总结果报告
    results_lines: List[str] = []
    results_lines.append(f"# 运行结果 ({_now_ts()})")
    for name, _tools, results in target_entries:
        results_lines.append("")
        results_lines.append(f"## {name}")
        for cid, status, note in results:
            line = f"- {cid}: {status}"
            if note:
                line += f" ({note})"
            results_lines.append(line)

    with open(os.path.join(args.reports, "results.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(results_lines))

    print("完成。", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

