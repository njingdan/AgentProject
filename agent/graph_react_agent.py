import operator
from typing import TypedDict, Annotated, Sequence
from langgraph.prebuilt import create_react_agent

from langchain.agents.middleware import ModelRequest
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import Tool
from langgraph.checkpoint.memory import MemorySaver

# 导入你的依赖（保持不变）
from agent.tools.agent_tools import (rag_summarize, get_user_location, get_user_id,
                                     get_current_month, fetch_external_data, fill_context_for_report)
from agent.tools.middleware import log_before_model, report_prompt_switch, monitor_tool
from mcp_client.base import mcp_weather_tool
from model.factory import chat_model
from utils.prompt_loader import load_system_prompts, load_report_prompts
from utils.logger_handler import logger


# 1. 定义LangGraph状态（仅保留核心字段）
class AgentState(TypedDict):
    messages: Annotated[Sequence[HumanMessage | AIMessage | ToolMessage], operator.add]


# 2. 工具包装（仅做格式兼容）
def _wrap_tools_for_langgraph():
    tools = [
        rag_summarize, get_user_location, get_user_id,
        get_current_month, fetch_external_data, fill_context_for_report, mcp_weather_tool
    ]
    langgraph_tools = []
    for tool in tools:
        if not isinstance(tool, Tool):
            langgraph_tools.append(
                Tool(name=tool.name, description=tool.description, func=tool.func)
            )
        else:
            langgraph_tools.append(tool)
    return langgraph_tools


class ReactAgent:
    def __init__(self):
        # 加载提示词
        self.system_prompt = load_system_prompts()

        # 获取LangGraph兼容的工具列表
        self.tools = _wrap_tools_for_langgraph()

        # 构建ReAct Agent的Prompt模板（复用你的格式）
        prompt_template = """
{system_prompt}

### 可用工具列表：
{tool_names}

### 工具详细说明：
{tools}

### 会话历史：
{history}

### 用户当前问题：
{input}

### 思考与行动：
{agent_scratchpad}
        """.strip()

        # 拼接工具信息（静态参数，仅初始化一次）
        tool_names = ", ".join([t.name for t in self.tools])
        tool_descs = "\n\n".join([f"{t.name}：{t.description}" for t in self.tools])

        # 核心：新版create_react_agent（直接返回已编译的图，无需二次compile）
        self.graph = create_react_agent(
            model=chat_model,
            tools=self.tools,
            prompt=prompt_template.format(
                system_prompt=self.system_prompt,
                tool_names=tool_names,
                tools=tool_descs,
                history="{history}",
                input="{input}",
                agent_scratchpad="{agent_scratchpad}"
            ),
            # 新版：中间件直接传入create_react_agent
            middleware=[
                monitor_tool,       # 自动拦截所有工具调用
                log_before_model,   # 自动在模型调用前执行
                report_prompt_switch# 自动在提示词生成前切换
            ],
            # 新版：会话历史通过checkpointer参数传入
            checkpointer=MemorySaver()
        )

    def execute_invoke(self, query: str, config: dict):
        """同步执行（极简版）"""
        # LangGraph原生入参格式：messages是核心字段
        inputs = {
            "messages": [HumanMessage(content=query)]
        }
        # 执行Agent（config传入thread_id，对应会话ID）
        result = self.graph.invoke(inputs, config=config)

        # 提取最终回答（复用你的Final Answer解析逻辑）
        final_answer = "我不知道"
        for msg in result["messages"]:
            if isinstance(msg, AIMessage) and "Final Answer:" in msg.content:
                final_answer = msg.content.split("Final Answer:")[-1].strip()
                break

        logger.info(f"[执行结果] 最终回答：{final_answer}")
        return final_answer

    def execute_stream(self, query: str, config: dict):
        """流式执行（极简版）"""
        inputs = {
            "messages": [HumanMessage(content=query)]
        }
        # 流式输出
        for chunk in self.graph.stream(inputs, config=config, stream_mode="values"):
            if "messages" in chunk and len(chunk["messages"]) > 0:
                last_msg = chunk["messages"][-1]
                if isinstance(last_msg, AIMessage) and last_msg.content:
                    yield last_msg.content.strip() + "\n"


# 测试代码（和你原有逻辑一致）
if __name__ == '__main__':
    agent = ReactAgent()
    config = {
        "configurable": {
            "thread_id": "user001",  # LangGraph标准会话ID字段
        }
    }

    # 测试1：工具调用场景
    print("===== 测试工具调用 =====")
    res = agent.execute_invoke("临猗今天的天气如何？", config=config)
    print("最终回答：", res)

    # # 测试2：报告生成场景（触发fill_context_for_report+提示词切换）
    # print("\n===== 测试报告生成 =====")
    # res2 = agent.execute_invoke("生成我的2026-02使用报告", config=config)
    # print("最终回答：", res2)