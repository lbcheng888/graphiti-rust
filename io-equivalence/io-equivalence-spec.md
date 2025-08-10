# Graphiti MCP I/O 等价性规范（Draft v0.1）

目的
- 明确 graphiti-rust MCP 与 Python 版 MCP 在“工具列表、工具调用、错误处理、传输模式（HTTP/stdio）、可选字段与容错行为”等方面的 I/O 等价性定义与验收标准。
- 为自动化对比脚手架提供规范化与判定依据。

范围
- 传输协议：MCP JSON-RPC 2.0 over HTTP 和/或 stdio
- 方法覆盖：initialize, initialized, tools/list, tools/call（涵盖全部工具），以及可选 resources/prompts（若 Python 版本提供）
- 端到端：请求入参（arguments/schema 默认值解析）、响应结构（jsonrpc/id/result 或 error）、工具输出的内容字段（content/text 等）
- 非功能：错误码/信息、缺省值行为、顺序稳定性、时间与浮点近似、UUID/路径/环境差异

术语
- 等价：在应用可接受的容差与归一化规则后，两边输出结构与语义一致，可互换用于上游客户端，不引入破坏性兼容性。
- 归一化（Normalization）：对不影响语义的差异做标准化处理（如 UUID、时间戳、顺序、浮点等）。

等价性判定规则
1) 协议层
- 必须返回 jsonrpc: "2.0"
- id 传递规则一致；通知（如 initialized）允许返回 result=null
- 错误响应必须包含 error.code 与 error.message；允许 error.data 差异（见错误映射）

2) 工具清单（tools/list）
- 工具集合应含有相同名称与语义的工具；若名称不同但语义一致，允许在映射表中声明别名（见 mapping.tools）
- inputSchema 属性需等价，并做结构化 diff：
  - 对比 type、required 列表一致；
  - 对 properties 中每个键：名称一致；type/enum/default/description 一致；
  - 可忽略纯文案差异（描述中的非语义性措辞不同），但必须保证默认值解析结果一致；
  - 允许一端存在额外可选属性，但在“严格模式”下将计为差异；“宽松模式”可豁免（见下文）。

3) 工具调用（tools/call）
- 输入：arguments 传参与默认值解析一致；缺省字段的默认值行为一致。
- 输出断言：
  - 结构化：对响应字段做“字段存在性与类型断言”，键名与层级必须一致；忽略字段顺序；
  - 非确定字段忽略：时间戳/UUID/路径/宿主信息等；
  - 提供严格与宽松两种模式（默认严格）：
    - 严格：键集合、类型、必填字段完全一致；多余字段计差异；
    - 宽松：允许多余可选字段，允许数值在容差范围内波动，允许文本在规范化后比对。

4) 错误与边界条件
- 相同输入导致的错误类别一致（4xx 参数错误 vs 5xx 服务内部错误）。
- 错误码映射：允许不同实现使用不同 code 值，但必须可映射到共同语义集（BadRequest, NotFound, Internal, NotImplemented, Timeout）。
- 可选字段缺失、类型不匹配、超限（limit）、非法 UUID、空查询等边界条件行为一致。

5) 非稳定字段归一化
- UUID：用占位符 {{UUID_n}} 归一化，并保持同一响应内的同值同占位。
- 时间戳：解析为 ISO8601 Z，再统一到秒级（或配置），如无法解析则原样。
- 浮点：四舍五入到 1e-6（可配置）；向量允许余弦相似度≥1-1e-3 判定等价。
- 顺序：对集合型字段按稳定 key 排序（若无 key，按字符串化后排序）；语义上无序的集合必须忽略顺序差。
- 文本：统一换行、去除前后空白、移除纯装饰性前缀（如“✅/❌/📊/⚠️ 等”）。
- 路径/URL：根据配置进行根路径重写或忽略大小写（Windows）。

6) 环境影响
- LLM/Embedding 非确定性：比较时允许在“数量、置信度、排序”上设容差；如属可选功能且另一路径存在纯规则实现（NER 规则），应优先选择确定性路径或锁定模型/随机种子。

覆盖矩阵（初版）
- initialize/initialized
- tools/list（列出全部工具并比对 schema）
- tools/call：
  - add_memory（Python 对齐 Rust add_memory）
  - search_memory_nodes（Python）↔ search_memory（Rust，若返回结构化）
  - search_memory_facts（Python，Rust 暂缺：部分等价）
  - get_episodes（Python，无 Rust 等价：非强制）
  - get_related_memories（仅 Rust：扩展能力）
  - add_code_entity / search_code / record_activity / batch_* / scan_project（Rust 扩展能力）

验收标准
- 针对上述每个方法与主要边界条件，自动化对比通过：
  - 差异级别为“非破坏性”的（仅文本/格式化差异）需被归一化消除；
  - 结构与语义差异为 0（或全部列入 remediation-plan 并给出修复建议与用例）。
- 生成以下产出：
  - artifacts/raw 与 artifacts/norm：保存原始与归一化后的 I/O；
  - reports/diff-report.md：按用例列出仍存在的实质差异；
  - reports/remediation-plan.md：按 P0/P1/P2 优先级给出修复建议与验收用例；
  - reports/conclusion.md：是否可无缝替换、必要前提、验收清单。

配置与可变项
- 传输：rust: {http|stdio}，python: {http|stdio}
- 端点：HTTP URL 或可执行命令（stdio）。
- 归一化策略：浮点精度、时间精度、忽略键集合、别名映射。
- 工具名映射与字段别名映射。

限制
- 本规范不要求修改任一工具实现；如发现不等价，以计划与用例形式输出并跟踪至验收通过。

版本与演进
- v0.1：初稿，覆盖主要路径与规则。
- v0.2+：基于实际对比结果，补充特定工具的结构化输出对齐细则与扩展边界条件。

## 风险与缓解（评测期）
- Python 端默认依赖远端嵌入或推理
  - 缓解：评测时设置占位 key，仅用于协议验证；在用例层面优先选择不触发推理/嵌入的参数；若无法避免，标注该用例为“部分等价”并可跳过计入结论。
- 模型本地路径与格式
  - 缓解：通过 crates/graphiti-mcp/config.free.toml 配置占位的本地模型路径与后端，执行前替换为真实可用的本地模型与格式（遵循“prefer_local_only=true”）。
- schema 差异来源
  - 缓解：diff/对比工具在宽松模式下仅检查与用例断言相关的字段，允许额外可选字段存在；严格模式仍会全量比对并报告差异。

