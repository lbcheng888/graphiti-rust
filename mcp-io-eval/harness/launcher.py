from __future__ import annotations

import os
import re
import socket
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, IO

from .config import TargetConfig


@dataclass
class RunningProcess:
    proc: subprocess.Popen
    stdout: IO[bytes]
    stderr: IO[bytes]
    assigned_port: Optional[int]
    log_thread: Optional[threading.Thread]
    stderr_path: Path


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start_target(t: TargetConfig, logs_dir: Path) -> RunningProcess:
    env = os.environ.copy()
    env.update(t.env or {})

    assigned_port: Optional[int] = None
    if t.needs_port:
        assigned_port = find_free_port()
        env[t.port_env_key] = str(assigned_port)

    stderr_path = logs_dir / f"{t.name}.stderr.log"
    stderr_f = open(stderr_path, "wb")

    proc = subprocess.Popen(
        t.cmd,
        cwd=t.cwd or None,
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=stderr_f,
        bufsize=0,
    )

    # 可选：等待服务 ready（通过 stderr/日志匹配）
    log_thread: Optional[threading.Thread] = None
    if t.ready_regex:
        ready_re = re.compile(t.ready_regex)
        # 后台读取 stderr 文件，直到匹配
        def wait_ready():
            start = time.time()
            while True:
                time.sleep(0.05)
                try:
                    text = stderr_path.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    text = ""
                if ready_re.search(text):
                    break
                if time.time() - start > t.timeout_sec:
                    break
        log_thread = threading.Thread(target=wait_ready, daemon=True)
        log_thread.start()

    assert proc.stdin and proc.stdout
    return RunningProcess(proc=proc, stdout=proc.stdout, stderr=stderr_f, assigned_port=assigned_port, log_thread=log_thread, stderr_path=stderr_path)


def stop_target(r: RunningProcess, timeout_sec: int = 10) -> Tuple[int, Optional[str]]:
    try:
        r.proc.stdin.close()  # 关闭输入，提示进程退出
    except Exception:
        pass
    try:
        code = r.proc.wait(timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        r.proc.terminate()
        try:
            code = r.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            r.proc.kill()
            code = r.proc.wait()
    try:
        r.stderr.close()
    except Exception:
        pass
    try:
        err_txt = r.stderr_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        err_txt = None
    return code, err_txt
