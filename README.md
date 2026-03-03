# AgentProject

> 智扫通机器人 · 智能 Agent Demo（大模型 + 工具调用 + RAG 知识库）

## 一、项目简介

AgentProject 是一个基于大语言模型的 **智能体（Agent）示例项目**。

项目目标：
- 用对话的方式控制/查询业务（例如智扫通相关场景）
- 通过 **工具（Tools）** 调用外部能力（搜索、天气等）
- 通过 **RAG 知识库** 使用本地文档进行问答和总结
- 支持 **会话记忆**，实现多轮对话

当前实现包括：
- ReAct 风格的 Agent（`agent/react_agent.py`、`agent/graph_react_agent.py`）
- Chroma 向量库 + 自定义 RAG 服务（`rag/`）
- MCP（Model Context Protocol）客户端工具（`mcp_client/`）
- Prompt 管理与加载（`prompts/` + `utils/prompt_loader.py`）

---

## 二、主要功能

### 1. 对话与 Agent 决策

- 入口脚本：`app.py`
- 通过大模型 + Prompt 实现一个智能助手：
  - 先分析用户的自然语言问题
  - 决定是否需要调用工具或知识库
  - 汇总最终答案返回给用户

核心 Agent 实现在：
- `agent/react_agent.py`：ReAct 模式的智能体
- `agent/graph_react_agent.py`：基于“节点/步骤”的图式 Agent（可选）

### 2. 工具（Tools）系统

工具相关代码在 `agent/tools/` 目录中：
- 封装了时间、用户信息等基础上下文工具
- 支持调用外部 MCP 服务（搜索、天气等）
- 可以根据业务扩展自己的工具（如：机器人状态查询、报表生成等）

MCP 客户端在 `mcp_client/` 目录中，例如：
- `mcp_bailian_search.py`：搜索相关工具
- `mcp_weather.py`：天气查询工具

### 3. RAG 知识库

RAG（Retrieval-Augmented Generation）相关代码在 `rag/` 目录：
- `vector_store.py`：向量库封装（基于 Chroma）
- `rag_service.py`：对外提供检索 + 生成的组合能力
- `rag/chroma_db/`：向量数据存储位置

典型流程：
1. 将本地文档转为向量，写入 Chroma
2. 用户提问时，先在向量库检索相关片段
3. 将检索结果 + 问题一起交给大模型，让模型在“有依据”的前提下回答

### 4. Prompt 管理与会话记忆

- Prompt 文件：`prompts/`
  - `main_prompt.txt`：主系统提示词
  - `report_prompt.txt`：生成报告/总结类任务的提示词
  - `rag_summarize.txt`：对检索结果进行总结的提示词
- Prompt 加载工具：`utils/prompt_loader.py`

会话与历史记录相关：
- `utils/file_history_store.py`：将聊天历史落地到文件
- `chat_history/`：存放历史对话记录

---

## 三、项目结构

```bash
AgentProject/
├── app.py                 # 项目入口（启动智能 Agent）
├── .env                   # 本地环境配置（API Key 等，不要提交到公开仓库）
├── agent/
│   ├── react_agent.py     # ReAct 智能体
│   ├── graph_react_agent.py  # 图式 Agent（可选）
│   ├── tools/             # 各类工具封装
│   ├── chat_history/      #（内部使用）聊天记录目录
│   └── chroma_db/         #（内部使用）向量库存储
├── rag/
│   ├── vector_store.py    # 向量库封装
│   ├── rag_service.py     # RAG 检索 + 生成服务
│   └── chroma_db/         # RAG 用的向量数据
├── prompts/
│   ├── main_prompt.txt    # 主系统 Prompt
│   ├── report_prompt.txt  # 报告类 Prompt
│   └── rag_summarize.txt  # RAG 总结 Prompt
├── mcp_client/
│   ├── base.py            # MCP 客户端基类
│   ├── mcp_bailian_search.py  # 搜索工具
│   └── mcp_weather.py     # 天气工具
├── utils/
│   ├── config_handler.py  # 读取配置
│   ├── file_handler.py    # 文件读写工具
│   ├── file_history_store.py # 聊天历史存储
│   ├── logger_handler.py  # 日志工具
│   └── path_tool.py       # 路径工具
├── config/                # 配置文件目录
├── data/                  # 业务数据/文档目录
├── logs/                  # 日志输出目录
├── chat_history/          # 全局聊天历史
├── chroma_db/             # 全局向量数据库
├── md5.text               # 文件校验或版本记录
└── test_env.py            # 环境测试脚本
```

---

## 四、环境配置与运行

### 1. 准备环境

- Python 版本推荐：**3.10+**
- 安装依赖（示例）：

```bash
pip install -r requirements.txt
```

> 如果暂时没有 `requirements.txt`，可以根据实际使用的库自行创建。

### 2. 配置 `.env`

在项目根目录创建或修改 `.env` 文件，例如：

```env
# 示例，按实际服务调整
DASHSCOPE_API_KEY=your_dashscope_key_here
OPENAI_API_KEY=your_openai_key_here
# 其他模型 / 服务的 Key...
```

不要将真实的密钥提交到公共仓库。

### 3. 启动项目

```bash
python app.py
```

如果使用的是 Streamlit 或其他 Web 框架，请根据实际代码改成：

```bash
streamlit run app.py
```

启动后，按照终端提示在浏览器中打开对应地址，即可与智能 Agent 对话。

---

## 五、如何扩展

1. **新增工具（Tools）**
   - 在 `agent/tools/` 中新增一个 Python 文件或函数
   - 在 Agent 初始化时注册该工具
   - 通过 Prompt 告诉模型这个工具的用途

2. **扩展 RAG 知识库**
   - 将新的文档放入 `data/` 目录
   - 编写脚本使用 `rag/vector_store.py` 将文档写入向量库
   - 在对话流程中调用 `rag_service.py` 进行检索

3. **自定义业务场景**
   - 修改 `prompts/main_prompt.txt` 或 `report_prompt.txt`，加入你的业务描述
   - 在工具层接入业务接口（如：机器人状态查询、任务下发、报表生成等）

---

## 六、注意事项

- `.env`、日志、向量数据库等文件包含敏感信息/大体量数据，**不建议提交到公开仓库**。
- 如果在使用过程中遇到报错，可以先查看：
  - `logs/` 目录下的日志
  - `test_env.py` 的输出（用于排查环境变量和依赖是否正常）。


如需对 README 进行进一步个性化（比如详细写智扫通具体业务流程、接口说明等），可以在本文件基础上继续补充对应章节。