仓库地址：<PUBLIC_REPOSITORY_URL>（提交前替换本占位符）

运行环境：Python 3.11+，支持 macOS/Linux。执行 python3 -m venv .venv，激活环境后运行 python -m pip install .。在本机终端设置 LLM_API_KEY、LLM_BASE_URL、LLM_MODEL；真实凭据不得写入文件。先运行 localloop doctor 检查模型及原生工具调用，再用 localloop run "编程任务" --workspace 项目目录启动；中断后可通过 --resume 会话编号继续。

特色功能：项目不使用任何 Agent 框架，独立实现模型-工具循环、消息历史、确定性上下文压缩、JSONL 会话恢复、错误回填、重试及多重终止条件。五个本地工具支持目录浏览、分段读文件、代码搜索、原子写文件和无 shell 命令执行；路径限制在工作区内，敏感文件被拦截，更新文件需匹配最近读取的 SHA-256，写入和命令默认需人工确认，子进程环境会移除密钥类变量。测试使用 FakeProvider，不消耗 API，覆盖正常流程、异常恢复和安全边界。

局限：当前仅处理 UTF-8 文本，安全策略用于降低误操作风险，并非操作系统级沙箱。
