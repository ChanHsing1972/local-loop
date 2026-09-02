# LocalLoop

Git 仓库地址：https://github.com/ChanHsing1972/local-loop

LocalLoop 是一个只支持交互模式的轻量级 coding agent。它不依赖 Agent 框架或服务端代码执行工具，通过 OpenAI 兼容 Chat Completions、原生 tool calling 和本地 Python 代码完成消息编排、工具执行与循环控制。

## 运行方式

要求 Python 3.11 或更高版本。在仓库根目录创建虚拟环境并安装项目：

```bash
python -m venv .venv
source .venv/bin/activate
pip install .
```

通过环境变量或仓库根目录的 `.env` 配置 `LLM_API_KEY`、`LLM_BASE_URL`、`LLM_MODEL`，然后在希望 Agent 操作的项目目录运行：

```bash
localloop
```

## 特色功能

- Agent 自主实现“模型推理→工具调用→结果回填→继续推理”的执行闭环
- 提供目录浏览、文件读取、全文搜索、文件写入和本地命令执行五类工具
- 支持流式输出、会话持久化与恢复、确定性上下文压缩、基于步数/时间/重复调用的终止条件、接口重试
- 写入已有文件前使用 SHA-256 检查读取版本，并展示 diff，请求用户批准
- 支持用户显式维护工作区记忆，记忆只在新会话创建时加载，旧会话保留原始快照
- 每次任务自动记录 `write_file` 修改前的文件检查点，可经反向 diff、哈希检查和批准后撤销
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
| `/remember TEXT` | 保存一条工作区记忆，从下一个新会话开始生效 |
| `/memory` | 查看当前工作区的有效记忆 |
| `/forget ID` | 遗忘指定工作区记忆 |
| `/checkpoints` | 查看最近的文件写入检查点 |
| `/undo` | 撤销最近一次由 `write_file` 产生的文件修改 |
| `/models` | 从兼容网关查询可用模型 |
| `/model ID` | 为当前交互进程切换模型 |
| `/approval ask` | 写文件和运行命令前逐次确认 |
| `/approval auto` | 自动批准写入和命令，仅限可信工作区 |
| `/clear` | 清屏并重新显示启动信息 |
| `/exit` | 退出程序 |

## 实现结构

- `agent.py`：Agent 核心执行引擎，负责模型/工具循环、工具结果回填、终止条件判断及会话创建与恢复
- `interactive.py`：交互式终端界面，负责持续会话、流式输出、斜杠命令、模型切换与审批模式切换
- `cli.py`：唯一程序入口，读取当前目录配置并启动交互界面
- `provider.py`：负责流式 Chat Completions 请求、tool call 拼装、错误重试和模型列表探测
- `tools.py`：定义五个本地工具、危险操作审批及路径、写入和命令安全限制
- `context.py`：上下文管理器，在超过预算时对较早消息及工具交互进行确定性压缩，同时保留最近上下文
- `session.py`：会话持久化模块，使用带版本号的仅追加 JSONL 事件记录保存消息和元数据，并支持历史会话恢复
- `memory.py`：保存用户显式提供的工作区记忆，并生成新会话使用的有限记忆快照
- `checkpoint.py`：记录任务内文件写入前的内容，并在哈希未发生冲突时安全恢复
- `config.py`：读取环境变量和 `.env`，校验 API 地址、模型与当前工作区
- `types.py`：模块间共享的消息、结果和接口类型