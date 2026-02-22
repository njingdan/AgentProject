import os
import asyncio
from langchain.agents import create_agent
from langchain_core.tools import Tool
from langchain_community.chat_models import ChatTongyi
from langchain_mcp_adapters.client import MultiServerMCPClient
from utils.logger_handler import logger

from mcp_client.base import mcp_client

def get_mcp_weather_sync(area: str) -> str:
    """
    同步调用 MCP 天气工具（适配 LangChain Agent 的同步调用）
    :param area: 城市名，如"天津"、"北京"
    :return: 格式化的天气信息
    """
    # 第一步：定义内部异步函数（复用原有的异步逻辑）
    async def async_inner():
        try:
            # 步骤1：获取 MCP 天气工具列表
            mcp_tools = await mcp_client.get_tools()
            # 步骤2：找到核心天气查询工具
            weather_tool = None
            for tool in mcp_tools:
                if "地名查询实时天气和预报" in tool.name:
                    weather_tool = tool
                    break
            if not weather_tool:
                return f"❌ 未找到天气查询工具"

            # 步骤3：调用工具查询指定城市天气
            tool_result = await weather_tool.ainvoke({"area": area})
            logger.info(f"✅ {area}天气查询结果：{tool_result}")
            # 步骤4：格式化结果
            return f"✅ {area}天气查询结果：{tool_result}"

        except Exception as e:
            return f"❌ 天气查询失败：{str(e)}"

    # 第二步：用asyncio.run执行异步逻辑，转为同步阻塞调用
    try:
        return asyncio.run(async_inner())
    except Exception as e:
        # 捕获asyncio执行过程中的异常（如事件循环问题）
        logger.error(f"❌ 同步执行天气查询失败：{str(e)}")
        return f"❌ 天气查询失败：{str(e)}"

mcp_weather_tool = Tool(
    name="mcp_weather_tool",
    func=get_mcp_weather_sync,
    description="查询指定城市的实时/未来天气，参数为城市名（如天津、北京），返回气温、湿度、风向等详细信息，优先使用该工具回答天气问题",
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
    agent = create_agent(model, [mcp_weather_tool])  # 注意：传工具列表（你的原代码漏了[]）

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