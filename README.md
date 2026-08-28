# LocalLoop

Git 仓库地址：https://github.com/ChanHsing1972/local-loop

LocalLoop 是一个从零实现的轻量级 coding agent，不依赖 LangChain、OpenAI Agents SDK 等 Agent 框架，也不使用服务端托管的代码执行或文件工具。程序通过 OpenAI 兼容 Chat Completions 与模型原生 tool calling 接口驱动模型，自行完成消息编排、工具执行和循环控制。

## 运行方式

要求 Python 3.11 或更高版本。在仓库根目录创建虚拟环境并安装项目：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

随后通过环境变量或 .env 配置 `LLM_API_KEY`、`LLM_BASE_URL`、`LLM_MODEL`。可先运行 `localloop doctor` 检查模型和原生工具调用，再运行 `localloop` 进入交互模式。也可使用 `localloop run "任务描述"` 执行一次性任务，`--workspace`/`-C` 指定待操作项目目录。

## 特色功能

- Agent 自主实现“模型推理→工具调用→结果回填→继续推理”的执行闭环
- 提供目录浏览、文件读取、全文搜索、文件写入和本地命令执行五类工具
- 支持流式输出、会话持久化与恢复、确定性上下文压缩、基于步数/时间/重复调用的终止条件、接口重试
- 写入已有文件前使用 SHA-256 检查读取版本，并展示 diff，请求用户批准
- 工具限制敏感路径、过滤凭据环境变量并阻止部分高风险命令

## 交互命令

| 命令 | 作用 |
| --- | --- |
| `/help` | 显示交互命令帮助 |
| `/status` | 显示模型、工作区、审批模式和当前会话 |
| `/new` | 清空当前上下文，下一条输入创建新会话 |
| `/resume ID` | 恢复指定历史会话 |
| `/sessions` | 列出当前工作区最近十个会话 |
| `/delete ID` | 确认后删除指定会话 |
| `/models` | 从兼容网关查询可用模型 |
| `/model ID` | 为当前交互进程切换模型 |
| `/approval ask` | 写文件和运行命令前逐次确认 |
| `/approval auto` | 自动批准写入和命令，仅限可信工作区 |
| `/clear` | 清屏并重新显示启动信息 |
| `/exit` | 退出程序 |

## 实现结构

- `agent.py`：Agent 核心执行引擎，负责模型/工具循环、工具结果回填、终止条件判断及会话创建与恢复
- `interactive.py`：交互式终端界面，负责持续会话、流式输出、斜杠命令、模型切换与审批模式切换
- `cli.py`：命令行程序入口，提供默认交互模式、`doctor` 环境诊断和 `run` 单任务执行模式
- `provider.py`：模型接口适配层，负责 OpenAI 兼容 Chat Completions 请求、普通及流式响应解析、tool call 拼装、错误重试和模型列表探测
- `tools.py`：本地工具系统，定义工具 Schema，并实现目录浏览、文件读取、全文搜索、文件写入和命令执行及相应的安全限制
- `context.py`：上下文管理器，在超过预算时对较早消息及工具交互进行确定性压缩，同时保留最近上下文
- `session.py`：会话持久化模块，使用带版本号的仅追加 JSONL 事件记录保存消息和元数据，并支持历史会话恢复
- `config.py`：运行配置模块，负责读取环境变量和 `.env`，校验 API 地址、模型、工作区及运行限制等配置
- `policy.py`：操作审批策略，负责写文件和运行命令前的用户确认，并提供自动批准及测试用批准策略
- `types.py`：公共数据结构与接口定义，包含 Agent 配置、ToolCall、ToolResult、AssistantTurn、运行状态以及 Provider、ApprovalPolicy 等协议
