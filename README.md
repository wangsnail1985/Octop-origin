# qingshan-data — Octop 用户数据快照（2026-11-28）

青山经科局实例的用户数据，供异地测试。基线：octop-qingai @ 87999e0c。

## 恢复方法（异地机器）

1. 按 octop-qingai 分支 README 部署代码并构建前端（或直接复制整目录）
2. 将本分支的 `.octop-data/` 整个目录复制为目标 `OCTOP_HOME`（如 `E:/octop-origin/.octop-data`）
3. 启动：`OCTOP_HOME=<目录> .venv/Scripts/octop.exe run`（首次登录密码在 `.octop-data/octop-login.txt`）

## 内容

- `octop.db`：一致性快照（用户/角色/部门/专家/提供商含 API Key/存储后端/知识库配置）
- `agents/{main,KVEHJ7,Z0JN5H}`：专家 manifest、工作区、会话历史
- `plugins/kb-search`：kb 知识库插件及凭据配置
- `config.json`、`octop-login.txt`、`security/`、`knowledge/` 等

⚠️ 含真实 API Key 与登录密码，仅限私有仓库使用。
