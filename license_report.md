# 许可证扫描与合规报告（占位版本）

本报告根据任务要求生成，不执行实际扫描，仅提供可复现实操步骤与建议。最终产物：
- security_py.json（已生成：占位版本，含重现实操命令）
- security_rs.json（已生成：占位版本，含重现实操命令）
- license_report.md（本文档）

—

## 一、依赖漏洞扫描（不执行，仅记录命令）

请在干净环境中执行以下命令以复现扫描并产出 JSON 报告：

- Python（pip-audit）：
  - 命令（记录，勿在此处执行）：
    pip install pip-audit && pip-audit -r graphiti-py/requirements.txt -f json -o security_py.json || true
  - 说明：
    - 在当前仓库根目录运行。
    - pip-audit 会解析 graphiti-py/requirements.txt 并输出 JSON 至 security_py.json。
    - 通过“|| true”避免在发现问题时导致 CI 失败（可按需移除）。

- Rust（cargo audit）：
  - 命令（记录，勿在此处执行）：
    cargo install cargo-audit && (cd graphiti-rust && cargo audit -q -f json > ../security_rs.json || true)
  - 说明：
    - 在 graphiti-rust 子目录执行审计，并将 JSON 输出至仓库根目录 security_rs.json。
    - 使用 -q 静默非必要输出；“|| true”避免 CI 失败。

—

## 二、许可证扫描与策略合规

### 2.1 Python 许可证扫描（pip-licenses）

推荐在隔离环境中执行，避免污染全局依赖：

1) 使用虚拟环境（示例）：
   - python -m venv .venv && source .venv/bin/activate  （Windows 使用 .venv\\Scripts\\activate）
2) 安装项目依赖：
   - pip install -r graphiti-py/requirements.txt
3) 安装工具：
   - pip install pip-licenses
4) 生成报告（JSON 示例）：
   - pip-licenses --format=json --with-authors --with-urls --with-license-file > licenses_py.json
5) 生成可读表格（可选）：
   - pip-licenses --with-authors --with-urls --with-license-file --no-license-path

注意：pip-licenses 读取的是当前环境中已安装的包。请确保步骤 (2) 已完成且未混入额外依赖。

### 2.2 Rust 许可证扫描（cargo deny）

若已存在 cargo-deny 配置（deny.toml），可执行：

- 安装工具：
  - cargo install cargo-deny
- 执行检查（JSON 输出示例）：
  - (cd graphiti-rust && cargo deny check licenses --format json > ../licenses_rs.json || true)

若尚无策略文件，可参考最小示例（将其保存为 graphiti-rust/deny.toml 或 cargo-deny 默认位置）：

```toml
[licenses]
# 允许的许可证列表示例（按需调整）
allow = ["MIT", "Apache-2.0", "BSD-3-Clause", "BSD-2-Clause"]
# 若不允许未声明许可证：
unlicensed = "deny"

[licenses.private]
# 允许将私有 crates 视为已合规：
ignore = true
```

常用命令：
- cargo deny init        # 在项目中生成初始配置
- cargo deny check       # 运行所有内置检查（advisories、licenses、bans）
- cargo deny check licenses -d warnings  # 将许可证问题降级为 warning（示例）

—

## 三、CI/CD 集成建议

- 将上述命令加入 CI 任务，并将 JSON 输出作为工件（artifact）上传保存。
- 在主分支严格执行：
  - pip-audit 与 cargo audit 报告不应含高危/严重漏洞（或设定白名单）。
  - pip-licenses/cargo-deny 的许可证应满足策略（deny.toml）。
- 可在 PR 中展示差异（如对比上一次扫描结果）。

—

## 四、当前仓库交付物状态

- security_py.json：占位文件，记录了可复现命令，未执行扫描。
- security_rs.json：占位文件，记录了可复现命令，未执行扫描。
- license_report.md：本文档，包含执行指引与策略示例。

如需我实际执行扫描并填充真实数据，请告知，我将按上述流程在隔离环境完成并更新报告。

