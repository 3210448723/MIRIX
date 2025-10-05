![Mirix Logo](https://github.com/RenKoya1/MIRIX/raw/main/assets/logo.png)

## MIRIX - 多代理（Multi-Agent）高级记忆系统个人助理

一款能通过屏幕观察与自然对话持续构建个性化长期记忆的本地优先 AI 助理。

| 🌐 [官网](https://mirix.io) | 📚 [文档](https://docs.mirix.io) | 📄 [论文 / Paper](https://arxiv.org/abs/2507.07957) |
<!-- | [Twitter/X](https://twitter.com/mirix_ai) | [Discord](https://discord.gg/mirix) | -->

---

### 核心特性 🔥

- **多代理记忆体系**：六类专业记忆组件（核心、情节、语义、程序、资源、知识库）由专门 Agent 协同管理
- **屏幕活动追踪**：持续视觉捕获 + 智能归纳，转化为结构化记忆
- **本地优先隐私**：长期数据默认本地存储，用户可控隐私策略
- **高级检索**：PostgreSQL 原生 BM25 全文 + 向量相似度混合搜索
- **多模态输入**：文本 / 图像 / 语音 / 屏幕截图 无缝处理
- **易集成**：提供轻量 Python SDK 与后端调用接口

---

## 快速上手（Quick Start）

### 面向终端用户（End Users）
若你希望直接构建自己的个人记忆，请参考桌面应用快速安装指引：<br/>
[https://docs.mirix.io/getting-started/installation/#quick-installation-dmg](https://docs.mirix.io/getting-started/installation/#quick-installation-dmg)

### 面向开发者（Developers）
如果你希望将 Mirix 作为「记忆系统后端」集成到你的项目，请参考：[Backend Usage](https://docs.mirix.io/user-guide/backend-usage/)。核心示例：

```bash
git clone git@github.com:Mirix-AI/MIRIX.git
cd MIRIX

# 创建并激活虚拟环境
python -m venv mirix_env
source mirix_env/bin/activate  # Windows: mirix_env\Scripts\activate

pip install -r requirements.txt
```

然后运行：
```python
from mirix.agent import AgentWrapper

# 使用配置初始化代理
agent = AgentWrapper("./mirix/configs/mirix.yaml")

# 发送一条可写入记忆的文本
agent.send_message(
    message="The moon now has a president.",
    memorizing=True,
    force_absorb_content=True
)
```
更多用法详见文档 Backend Usage 部分。

---

## Python SDK（全新）🎉

使用更简洁的接口快速调用 Mirix 记忆能力：

### 安装
```bash
pip install mirix
```

### 快速示例
```python
from mirix import Mirix

# 初始化（默认使用 Google Gemini 2.0 Flash，可传入 api_key）
memory_agent = Mirix(api_key="your-google-api-key")

# 添加记忆
memory_agent.add("The moon now has a president")
memory_agent.add("John loves Italian food and is allergic to peanuts")

# 上下文聊天（自动引用记忆）
response = memory_agent.chat("Does the moon have a president?")
print(response)  # 例："Yes, according to my memory, the moon has a president."

response = memory_agent.chat("What does John like to eat?")
print(response)  # 例："John loves Italian food. However, he's allergic to peanuts."
```

---

## 许可（License）

本项目基于 Apache License 2.0 开源，详见 [LICENSE](LICENSE)。

---

## 后端本地运行环境配置（源码方式）🇨🇳

以下为在本地「非 Docker、非桌面打包」情况下启动 Mirix 后端的必备说明：

### 1. 基础依赖
- Python 3.10+（目前 3.12 兼容）
- pip / venv 或 conda（建议使用虚拟环境）
- Git
- （可选）Node.js + npm（若需运行前端界面）

### 2. 安装 PostgreSQL
Mirix 推荐使用 Postgres（支持全文 + 向量检索）。缺省条件下可回退 SQLite（功能受限且无向量搜索）。

Ubuntu/Debian 示例：
```bash
sudo apt-get update
sudo apt-get install -y postgresql postgresql-contrib
```
验证：
```bash
sudo systemctl status postgresql
psql -V  # 例：16.x
```

### 3. 创建数据库与用户
```bash
sudo -u postgres psql -c "CREATE DATABASE mirix;"
sudo -u postgres psql -c "CREATE ROLE mirix LOGIN PASSWORD 'mirix';"
```
如需重置：
```bash
sudo -u postgres psql -c "ALTER USER postgres WITH PASSWORD 'your_new_password';"
```

### 4. 安装/启用 pgvector 扩展
若出现 `type "vector" does not exist` 说明缺少扩展。
```bash
sudo apt-get install -y postgresql-16-pgvector || sudo apt-get install -y postgresql-pgvector
sudo -u postgres psql -d mirix -c "CREATE EXTENSION IF NOT EXISTS vector;"
psql -h 127.0.0.1 -U mirix -d mirix -c "\\dx" | grep vector || echo "未找到 vector 扩展"
```
若无包：
```bash
sudo apt-get install -y build-essential git postgresql-server-dev-$(psql -V | awk '{print $3}' | cut -d. -f1)
git clone https://github.com/pgvector/pgvector.git
cd pgvector && make && sudo make install
sudo -u postgres psql -d mirix -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

### 5. 环境变量 / `.env`
在项目根目录创建 `.env`：
```
MIRIX_PG_URI=postgresql+pg8000://mirix:mirix@127.0.0.1:5432/mirix
GEMINI_API_KEY=你的GoogleGeminiKey
```
或使用拆分字段：
```
MIRIX_PG_DB=mirix
MIRIX_PG_USER=mirix
MIRIX_PG_PASSWORD=mirix
MIRIX_PG_HOST=127.0.0.1
MIRIX_PG_PORT=5432
```
代码逻辑会优先使用完整 URI；缺失则拼接；都没有则回退默认：`postgresql+pg8000://mirix:mirix@localhost:5432/mirix`。

### 6. 安装依赖
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 7. 启动后端服务
```bash
python main.py --host 0.0.0.0 --port 47283
```
日志样例：
```
Creating engine postgresql+pg8000://...
[启动诊断] ...
```
若缺少 pgvector 会尝试自动 `CREATE EXTENSION IF NOT EXISTS vector;` 并给出提示。

### 8. 常见错误排查
| 现象 | 原因 | 解决 |
|------|------|------|
| password authentication failed | 密码不正确 / host 规则不允许 | 重设密码；检查 `pg_hba.conf` 里是否存在 `host all all 127.0.0.1/32 scram-sha-256` |
| AttributeError: 'NoneType' object has no attribute 'decode' | URI 中缺少密码 | 使用完整 `user:password` 形式 |
| type "vector" does not exist | 未安装或未启用 pgvector | 安装扩展并执行 `CREATE EXTENSION vector;` |
| could not connect to server | 服务未启动 / 端口错误 / 防火墙 | `systemctl status postgresql` & 检查 5432 |
| 表结构冲突 | 旧 schema 残留 | 备份后清理冲突表，当前无自动迁移 |
| SQLite schema 失效 | 不支持迁移 | 删除 `~/.mirix/sqlite.db` 或改用 Postgres |

### 9. 检查 pgvector 是否启用
```bash
psql -h 127.0.0.1 -U mirix -d mirix -c "SELECT extname, version FROM pg_extension WHERE extname='vector';"
```

### 10. 安全建议
- 不要提交真实 API Key
- 使用 `.env` + `.gitignore`
- 生产最小权限原则，不用超级用户运行

### 11. 快速诊断脚本
```bash
#!/usr/bin/env bash
set -e
echo "[1] 测试登录" && PGPASSWORD=mirix psql -h 127.0.0.1 -U mirix -d mirix -c "SELECT current_user;"
echo "[2] 检查扩展" && PGPASSWORD=mirix psql -h 127.0.0.1 -U mirix -d mirix -c "\\dx" | grep vector || echo "未安装 vector"
echo "[3] 测试建表" && PGPASSWORD=mirix psql -h 127.0.0.1 -U mirix -d mirix -c "CREATE TABLE IF NOT EXISTS _mirix_diag(id int);"
```

### 12. 使用 psycopg 驱动（可选）
```bash
pip install "psycopg[binary]"
export MIRIX_PG_URI=postgresql+psycopg://mirix:mirix@127.0.0.1:5432/mirix
```

### 13. 切换到 SQLite（临时 / 轻量场景）
不设置 `MIRIX_PG_URI` 及相关字段时自动回退：`~/.mirix/sqlite.db`。此模式无向量检索能力。

---
若某一步仍有问题，请附带第一段错误日志 + 关键信息提交 issue 以便协助。

---

## 社区与联系（Community）

问题 / 建议 / Bug：请提 Issue 或邮件：`yuwang@mirix.io`

### 💬 Discord 社区
实时讨论与支持：<br/>
**https://discord.gg/5HWyxJrh**

### 🎯 每周讨论会
内容：问题答疑 / 方向建议 / 记忆体系讨论 / 需求反馈  
时间（PST）：每周五晚 8-9 PM  
Zoom: https://ucsd.zoom.us/j/96278791276

### 📱 微信群
<div align="center">
<img src="frontend/public/wechat-qr.png" alt="WeChat QR Code" width="200"/><br/>
<strong>扫码加入 Mirix 微信群</strong>
</div>

---

## Star 历史

[![Star History Chart](https://api.star-history.com/svg?repos=Mirix-AI/MIRIX&type=Date)](https://star-history.com/#Mirix-AI/MIRIX.&Date)

---

## 致谢（Acknowledgement）
感谢 [Letta](https://github.com/letta-ai/letta) 开源其框架，为本项目记忆系统奠定基础。

---

如果你觉得本项目有价值，欢迎 Star ⭐ 支持 —— 这会极大鼓励我们持续改进记忆架构与检索能力。
