from typing import Callable

from langchain_core.runnables import RunnableConfig
from utils.prompt_loader import load_system_prompts, load_report_prompts
from langchain.agents.middleware import wrap_tool_call, before_model, dynamic_prompt, ModelRequest
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.runtime import Runtime
from langgraph.types import Command
from utils.logger_handler import logger


@wrap_tool_call
def monitor_tool(
        # 请求的数据封装
        request: ToolCallRequest,
        # 执行的函数本身
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
) -> ToolMessage | Command:             # 工具执行的监控
    logger.info(f"[tool monitor]执行工具：{request.tool_call['name']}")
    logger.info(f"[tool monitor]传入参数：{request.tool_call['args']}")

    try:
        result = handler(request)
        logger.info(f"[tool monitor]工具{request.tool_call['name']}调用成功")

        if request.tool_call['name'] == "fill_context_for_report":
            request.runtime.context["report"] = True

        return result
    except Exception as e:
        logger.error(f"工具{request.tool_call['name']}调用失败，原因：{str(e)}")
        raise e


@before_model
def log_before_model(state, runtime):
    messages = state.get("messages", [])

    if not messages:
        logger.debug("[log_before_model] 当前无messages（可能是v1 create_agent输入）")
        return state

    last_msg = messages[-1]
    logger.debug(
        f"[log_before_model]{type(last_msg).__name__} | {getattr(last_msg, 'content', '')}"
    )
    return state


@dynamic_prompt                 # 每一次在生成提示词之前，调用此函数
def report_prompt_switch(request:ModelRequest):     # 动态切换提示词
    # print(request.runtime)
    #
    is_report = request.runtime.context.get("report", False)
    if is_report:               # 是报告生成场景，返回报告生成提示词内容
        return load_report_prompts()

    return load_system_prompts()

