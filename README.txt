仓库地址：https://github.com/ChanHsing1972/local-loop

运行：需要 Python 3.11+。执行 python3 -m venv .venv，激活后运行 python -m pip install .；复制 .env.example 为 .env，填写 LLM_API_KEY、LLM_BASE_URL、LLM_MODEL（.env 已忽略，严禁提交真实凭据）。先运行 localloop doctor 验证接口；在目标目录运行 localloop，即可连续输入任务和追问。空闲时界面不预留菜单空白；输入 / 后才展示全部命令并支持筛选、选择、Tab 补全和 Enter 执行；/new 新建上下文，/resume 恢复并重绘历史对话，/delete 删除指定历史，/model 切换模型。

特色：项目不使用 Agent 框架，独立实现流式模型输出、原生工具调用解析、本地模型-工具循环、消息历史、确定性上下文压缩、JSONL 会话恢复、有界重试和多重终止条件。终端逐项显示实际文件、行号、命令、退出码和结果摘要，而非笼统步数。五个本地工具支持目录、读取、搜索、原子写入和无 Shell 命令；路径限制在工作区，敏感文件被拦截，更新需匹配读取时 SHA-256，写入和命令默认确认，子进程移除密钥变量。测试覆盖正常闭环、流式分片、空响应、断流、异常恢复和安全边界；另以真实 API 完成工具调用、结果回填及修复测试任务。局限：仅处理 UTF-8 文本，防误操作机制不是操作系统沙箱。
