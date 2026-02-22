import os
import asyncio
from langchain.agents import create_agent
from langchain_core.tools import Tool
from langchain_community.chat_models import ChatTongyi
from langchain_mcp_adapters.client import MultiServerMCPClient
from utils.logger_handler import logger
from mcp_client.base import mcp_client

def get_balian_sync(query: str) -> str:
    """
    同步调用 MCP 联网搜索工具（适配 LangChain Agent 的同步调用）
    :param query: 查询的问题
    :return: 返回的结果
    """
    # 第一步：定义内部异步函数（复用原有的异步逻辑）
    async def async_inner():
        try:
            # 步骤1：获取 MCP 天气工具列表
            mcp_tools = await mcp_client.get_tools()
            # 步骤2：找到核心联网查询工具
            web_tool = None
            for tool in mcp_tools:
                if "bailian_web_search" in tool.name:
                    web_tool = tool
                    break
            if not web_tool:
                return f"❌ 未找到联网查询工具"

            # 步骤3：调用工具查询指定城市天气
            tool_result = await web_tool.ainvoke({"query": query,"count":1})
            logger.info(f"✅ {query}联网查询结果：{tool_result}")
            # 步骤4：格式化结果
            return f"✅ {query}联网查询结果：{tool_result}"

        except Exception as e:
            return f"❌ 联网查询失败：{str(e)}"

    # 第二步：用asyncio.run执行异步逻辑，转为同步阻塞调用
    try:
        return asyncio.run(async_inner())
    except Exception as e:
        # 捕获asyncio执行过程中的异常（如事件循环问题）
        logger.error(f"❌ 同步执行联网查询失败：{str(e)}")
        return f"❌ 联网查询失败：{str(e)}"

mcp_bailian_web_search_tool = Tool(
    name="mcp_bailian_web_search_tool",
    func=get_balian_sync,
    description="基于通义实验室 Text-Embedding，GTE-reRank，Query 改写，搜索判定等多种检索模型及语义理解，串接专业搜索工程框架及各类型实时信息检索工具，提供实时互联网全栈信息检索，提升 LLM 回答准确性及时效性。"
)

async def main():
    # 1. 获取 MCP 天气工具
    tools = await mcp_client.get_tools()
    print(f"✅ 发现 MCP 工具：{[tool.name for tool in tools]}")

    # 2. 初始化通义千问大模型
    model = ChatTongyi(
        model="qwen3-max",
        api_key=os.getenv("DASHCOPE_API_KEY"),
        temperature=0.1
    )

    #
    agent = create_agent(model, [mcp_bailian_web_search_tool])  # 注意：传工具列表（你的原代码漏了[]）

    #
    result = agent.invoke({
        "messages":[
            {"role":"system","content":"你是位助手，需要调用工具帮助用户.给出我你调用工具的思考过程,严格符合ReAct范式！"},
            {"role":"user","content":"天津今天天气如何？明天呢？"},
        ]
    })

    # 5. 输出结果
    print("\n📝 最终回答：")
    for res in result['messages']:
        print(res.content)

if __name__ == "__main__":
    asyncio.run(main())