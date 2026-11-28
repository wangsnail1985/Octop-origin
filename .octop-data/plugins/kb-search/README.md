# 企业知识库检索（kb-search）

Octop 专家插件：把 **kb-service**（知识库双引擎服务：语义向量 + 内置 PageIndex 页码级
检索，RRF 融合 + 标签权限过滤）作为一个 `search_knowledge_base` 工具接入专家对话，
用于**替换** Octop 内置知识库的检索链路。

## 调用示例

```text
search_knowledge_base(query="智能体产业学院 培养方案", top_k=8)
search_knowledge_base(query="教务处 期末考试安排")
```

结果片段携带「文档名 + 页码/锚点 + 相关度 + 召回路（semantic/page/graph）」，
模型被要求在回答中标注来源，不虚构引用。

## 配置（kb-search.json，与本插件同目录，改动即时生效）

| 字段 | 说明 |
| --- | --- |
| `base_url` | kb-service 地址，默认 `http://127.0.0.1:8125` |
| `request_timeout_seconds` | HTTP 超时，默认 30 |
| `default_top_k` | 缺省返回片段数（工具参数 `top_k` 1–20） |
| `max_content_chars` | 单片段内容截断长度，默认 800 |
| `default_account` | 共享服务账户（所有映射都未命中时兜底） |
| `users` | **按用户名映射**：octop 用户名 → kb-service 账户 |
| `roles` | **按角色映射**：octop 角色名 → kb-service 账户（角色对应部门，推荐） |

### 身份与权限（角色 ↔ 部门）

kb-service 的标签权限（可读用户/所属部门）以**账户身份**为准（v4.1 安全要求）。
插件按以下顺序解析当前对话用户的 kb 账户：

1. **角色映射**：读 octop 用户在 octop.db 的 `role_name`（缓存 30s），命中 `roles` 表
   即用对应 kb 账户。例：octop 角色「教务处」→ kb 账户 `octop-jwc`（dept=学校教务处），
   只能看到部门标签为 学校教务处 的文档。
2. **用户名映射**：`users` 表按 octop 用户名直配。
3. **兜底**：`default_account`（svc-octop，dept=it——只能看到无部门标签的文档）。

新增部门三步：octop 建角色（用户管理→角色）→ kb-service 建对应部门账户
（`python -m kb.cli user add <名> --password <密> --dept <部门>`）→ `roles` 加一条映射。
令牌登录后缓存至到期前 60 秒，401 时自动重登一次再重试。

## 安装与启用

```bash
e:\octop\bin\octop.cmd plugin install e:\octop\plugins-src\kb-search
e:\octop\bin\octop.cmd plugin reload   # 离线重载；或重启 octop run
```

安装后 Octop 会把源目录复制到 `e:\octop\plugins\kb-search\`，**请编辑安装目录下的
`kb-search.json`**（而不是本源目录）。全局启用后专家默认获得该工具，可在各专家的
工具设置中单独关闭。

## 替换内置知识库的配套步骤

1. 停用内置知识库功能（管理端「知识库」能力开关，即 settings 表
   `knowledge_bases_enabled=false`），隐藏上传/检索 UI 并关闭自动注入；
2. 内部资料继续走 kb-service 的上游推送/同步链路（附录 B 契约），不经 Octop 上传。

## 依赖

- kb-service 单进程部署、同内网可达（需求文档 D7/D8）
- Octop venv 已含 `httpx`（arxiv-search 插件同款依赖）
