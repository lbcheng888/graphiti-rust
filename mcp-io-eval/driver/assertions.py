from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union
import json
import os


class JsonRpcError(AssertionError):
    pass


class ShapeAssertionError(AssertionError):
    pass


_DEFAULT_MAPPING_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "cases.mapping.json"
)


def _load_nondeterministic_fields(mapping_path: Optional[str]) -> Set[str]:
    path = mapping_path or _DEFAULT_MAPPING_PATH
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            fields = data.get("nondeterministic_fields", [])
            return {str(x) for x in fields}
    except Exception:
        return set()


def _strip_nondeterministic(obj: Any, fields: Set[str]) -> Any:
    """深拷贝式地移除 obj 中所有命中的非确定性字段（按键名匹配）。"""
    if not fields:
        return obj
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k in fields:
                continue
            out[k] = _strip_nondeterministic(v, fields)
        return out
    if isinstance(obj, list):
        return [_strip_nondeterministic(x, fields) for x in obj]
    return obj


def _typeof(x: Any) -> str:
    if isinstance(x, dict):
        return "object"
    if isinstance(x, list):
        return "array"
    if x is None:
        return "null"
    return type(x).__name__


def assert_jsonrpc_ok(resp: Dict[str, Any]):
    """
    断言 JSON-RPC 响应为成功：
    - 至少包含 jsonrpc=2.0 和 id
    - 不包含 error 字段
    - 包含 result 字段
    """
    if not isinstance(resp, dict):
        raise JsonRpcError(f"response not an object: {type(resp).__name__}")
    if resp.get("jsonrpc") != "2.0":
        raise JsonRpcError(f"jsonrpc version not 2.0: {resp.get('jsonrpc')}")
    if "id" not in resp:
        raise JsonRpcError("missing id")
    if "error" in resp:
        raise JsonRpcError(f"contains error: {resp['error']}")
    if "result" not in resp:
        raise JsonRpcError("missing result")


def _compare_types(expect: Any, actual: Any, path: str) -> None:
    te, ta = _typeof(expect), _typeof(actual)
    if te != ta:
        raise ShapeAssertionError(f"{path}: 类型不匹配 expect={te} actual={ta}")


def _compare_object(
    expect: Dict[str, Any],
    actual: Dict[str, Any],
    path: str,
    strict: bool,
) -> None:
    exp_keys = set(expect.keys())
    act_keys = set(actual.keys())
    missing = exp_keys - act_keys
    extra = act_keys - exp_keys
    if missing:
        raise ShapeAssertionError(f"{path}: 缺少字段 {sorted(missing)}")
    if strict and extra:
        raise ShapeAssertionError(f"{path}: 存在额外字段 {sorted(extra)}（严格模式）")
    for k in exp_keys:
        assert_shape(actual[k], expect[k], strict=strict, allow_array_or_object=False)


def _compare_array(
    expect: List[Any],
    actual: List[Any],
    path: str,
    strict: bool,
) -> None:
    # 仅比较首元素形状；若期望非空而实际为空则报错
    if expect and not actual:
        raise ShapeAssertionError(f"{path}: 数组为空但期望非空")
    if not expect:
        # 期望为空数组，对长度不做强约束
        return
    # 使用首元素作为样例进行形状比对
    e0 = expect[0]
    a0 = actual[0]
    assert_shape(a0, e0, strict=strict, allow_array_or_object=False)


def _maybe_align_array_object(actual: Any, expect: Any) -> Tuple[Any, Any]:
    # 允许对象/数组互容：当其中一侧为对象、另一侧为数组时进行对齐
    if isinstance(expect, dict) and isinstance(actual, list):
        # 用 actual[0] 与 expect 比对；若 actual 为空则使用空对象触发缺字段
        return (actual[0] if actual else {}, expect)
    if isinstance(expect, list) and isinstance(actual, dict):
        # 将 actual 包装成单元素数组来与 expect[0] 比对
        return ([actual], expect)
    return actual, expect


def assert_shape(
    actual: Any,
    expect: Any,
    *,
    strict: bool = True,
    nondeterministic_fields: Optional[Iterable[str]] = None,
    allow_array_or_object: bool = True,
    mapping_path: Optional[str] = None,
) -> None:
    """
    结构与类型断言：
    - 严格模式：要求字段存在且类型匹配；禁止额外字段
    - 宽松模式：允许额外字段；类型需匹配
    - 非确定性字段忽略（来自 cases.mapping.json 的 nondeterministic_fields）
    - 数组/对象互容（如 search 工具返回数组或对象）
    """
    nd_fields: Set[str] = set(nondeterministic_fields or []) or _load_nondeterministic_fields(mapping_path)

    # 预处理：移除非确定性字段
    actual_p = _strip_nondeterministic(actual, nd_fields)
    expect_p = _strip_nondeterministic(expect, nd_fields)

    # 数组/对象互容
    if allow_array_or_object:
        actual_p, expect_p = _maybe_align_array_object(actual_p, expect_p)

    # 顶层类型检查
    te, ta = _typeof(expect_p), _typeof(actual_p)
    if te != ta:
        # 若允许互容，上面已处理；否则报错
        raise ShapeAssertionError(f"$: 顶层类型不匹配 expect={te} actual={ta}")

    def _recur(a: Any, e: Any, p: str):
        if isinstance(e, dict):
            _compare_object(e, a, p, strict)
        elif isinstance(e, list):
            if not isinstance(a, list):
                raise ShapeAssertionError(f"{p}: 类型不匹配 expect=array actual={_typeof(a)}")
            _compare_array(e, a, p, strict)
        else:
            # 基本类型仅校验类型一致即可；若希望值相等，外部可单独断言
            if type(a) is not type(e):
                raise ShapeAssertionError(
                    f"{p}: 基本类型不匹配 expect={type(e).__name__} actual={type(a).__name__}"
                )

    _recur(actual_p, expect_p, "$")
