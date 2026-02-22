import os
import asyncio
from langchain.agents import create_agent
from langchain_core.tools import Tool
from langchain_community.chat_models import ChatTongyi
from langchain_mcp_adapters.client import MultiServerMCPClient
from utils.logger_handler import logger
import dotenv

# 加载环境变量
dotenv.load_dotenv()

# 配置 MCP 客户端
mcp_client = MultiServerMCPClient(
    {
        "WeatherSearch": {
            "transport": "streamable_http",
            "url": "https://dashscope.aliyuncs.com/api/v1/mcps/mcp-ZWU3ZmQ3OTVlMjY5/mcp",
            "headers": {"Authorization": f"Bearer {os.getenv('DASHSCOPE_API_KEY')}"},
        },
        "WebSearch": {
            "transport": "sse",
            "url": "https://dashscope.aliyuncs.com/api/v1/mcps/WebSearch/sse",
            "headers": {"Authorization": f"Bearer {os.getenv('DASHSCOPE_API_KEY')}"},
        },
    }
)

# 获取MCP 工具列表
async def get_mcp_tools():
    """异步获取 MCP 天气工具"""
    return await mcp_client.get_tools()


async def main():
    # 1. 获取 MCP 天气工具
    tools = await get_mcp_tools()
    print(f"✅ 发现 MCP 工具：{[tool.name for tool in tools]}")


if __name__ == "__main__":
    asyncio.run(main())