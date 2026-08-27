仓库地址：https://github.com/ChanHsing1972/local-loop

运行环境：Python 3.11+，支持 macOS/Linux。执行 python3 -m venv .venv，激活后运行 python -m pip install .。在本机终端设置 LLM_API_KEY、LLM_BASE_URL、LLM_MODEL，真实凭据不得写入文件。先运行 localloop doctor 检查模型和原生工具调用；进入目标目录执行 localloop，即可在交互界面连续输入任务和追问。/help 显示命令，/new 新建上下文，/resume 恢复会话，/model 切换模型，/approval 调整审批。自动化场景仍可使用 localloop run "任务" --workspace 项目目录。

特色功能：项目不使用任何 Agent 框架，独立实现交互式多轮会话、模型-工具循环、消息历史、确定性上下文压缩、JSONL 恢复、错误回填、重试和多重终止条件。五个本地工具支持目录浏览、分段读文件、代码搜索、原子写文件和无 Shell 命令执行；路径限制在工作区内，敏感文件被拦截，更新文件需匹配最近读取的 SHA-256，写入和命令默认需确认，子进程环境会移除密钥类变量。测试使用 FakeProvider，不消耗 API，覆盖正常流程、持续对话、异常恢复和安全边界。

局限：当前仅处理 UTF-8 文本，安全策略用于降低误操作风险，并非操作系统级沙箱。
