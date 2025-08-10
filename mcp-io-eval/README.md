# mcp-io-eval

构建自动化对齐验证脚手架（不改动被测工具代码）。

目录结构
- harness 通用启动器与 MCP 客户端调用器
- cases 用例矩阵
- artifacts 原始输出与规范化输出
- reports 报告与日志
- scripts 常用脚本（规范化、对比、汇总）
- README 复现说明

功能概述
- launcher
  - 以 JSON/YAML 声明两个被测目标（进程启动命令、工作目录、环境变量、超时时间）
  - 以子进程方式启动/停止，被测工具退出后清理
  - 若为端口型服务，启动前探测端口占用并注入随机空闲端口（环境变量或参数）
- mcp runner
  - 通过标准输入输出与被测工具建立 MCP JSON-RPC 会话
  - 执行 tools/list 获取工具函数签名并保存快照
  - 针对每条用例执行 tools/call，发送相同输入
  - 捕获标准输出/标准错误/退出码与会话日志
- canonicalizer 规范化器
  - 对 JSON 输出排序、去除易变字段、标准化空白与小数精度
  - 对流式输出合并重建最终结果，并保留原始事件轨迹
- differ 差异对比
  - 生成逐用例的结构差异与文本差异
  - 输出机器可读 JSON 与人可读 Markdown

快速开始
1) 编辑 cases/cases.yaml 定义用例。
2) 编辑 harness/targets.yaml 定义两个被测目标 target_a 与 target_b。
3) 运行:
   - python3 scripts/run_all.py
   - 结果：artifacts/ 下保存原始与规范化输出，reports/ 下生成差异与汇总报告。

注意
- 需要 Python 3.9+
- 若使用 YAML，请安装依赖：pip install pyyaml
- MCP 方法名默认使用 tools/list 与 tools/call（可在 harness/config.py 中调整）。

风险与缓解
- Python 端默认依赖远端嵌入或推理
  - 评测时为远端提供方设置占位 key（仅作协议/连通性验证，不实际发起推理），并在用例中优先选择不会触发推理/嵌入的参数组合；若无法避免，请将该用例标记为“部分等价”并在汇总时可选择跳过。
- 模型本地路径与格式
  - 使用 crates/graphiti-mcp/config.free.toml 中的占位路径作为模板，执行前替换为真实的本地模型路径与格式（例如 Candle + GGUF），同时在 defaults 中确保 prefer_local_only=true，避免任何远端调用。
- schema 差异来源
  - 对比阶段启用“宽松模式”：differ 仅检查与该用例断言相关的字段，允许额外可选字段存在；如需严格校验可开启严格模式以获得完整差异明细。

