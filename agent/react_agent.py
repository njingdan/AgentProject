from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableWithMessageHistory, RunnableLambda
from langchain_core.runnables.base import Runnable
from model.factory import chat_model
from utils.file_history_store import get_history
from utils.prompt_loader import load_system_prompts, load_report_prompts
from agent.tools.agent_tools import (rag_summarize, get_user_location, get_user_id,
                                     get_current_month, fetch_external_data, fill_context_for_report)
from mcp_client.mcp_weather import mcp_weather_tool
from mcp_client.mcp_bailian_search import mcp_bailian_web_search_tool
from utils.logger_handler import logger
from typing import Any, Dict
import threading

# ===================== 新增：会话上下文存储（线程安全） =====================
# 用字典存储会话标记，key=session_id，value=是否为报告场景
session_context = {}
# 线程锁，保证多请求下数据安全
context_lock = threading.Lock()

def set_session_report_flag(session_id: str, is_report: bool):
    """设置会话的报告场景标记"""
    with context_lock:
        session_context[session_id] = is_report

def get_session_report_flag(session_id: str) -> bool:
    """获取会话的报告场景标记"""
    with context_lock:
        return session_context.get(session_id, False)

def clear_session_report_flag(session_id: str):
    """清空会话的报告场景标记（避免污染下一次请求）"""
    with context_lock:
        if session_id in session_context:
            del session_context[session_id]


class ReactAgent:
    def __init__(self):
        # 1. 加载系统提示词（基础版，动态替换时会覆盖）
        system_prompt_str = load_system_prompts()

        # 2. 构建ReAct Agent标准格式的Prompt模板
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

        # 初始化Prompt（注意：这里不设置partial_variables的system_prompt，留空让动态逻辑填充）
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["input", "history", "agent_scratchpad", "system_prompt"],  # 新增system_prompt入参
            partial_variables={
                "tool_names": ", ".join([tool.name for tool in self._get_tools()]),
                "tools": "\n\n".join([f"{tool.name}：{tool.description}" for tool in self._get_tools()])
            }
        )

        # 3. 创建ReAct Agent核心
        agent_core = create_react_agent(
            llm=chat_model,
            tools=self._get_tools(),
            prompt=prompt,
        )

        # 4. 包装AgentExecutor（优化解析失败处理）
        self.base_agent = AgentExecutor(
            agent=agent_core,
            tools=self._get_tools(),
            verbose=True,
            handle_parsing_errors=self._custom_error_handler,
            max_iterations=5,
            return_intermediate_steps=True,
        )

        # 5. 包装监控逻辑（替代原中间件）
        self.base_agent = self._wrap_agent_with_monitor(self.base_agent)

        # 6. 包装会话历史
        self.agent = RunnableWithMessageHistory(
            self.base_agent,
            get_history,
            input_messages_key="input",
            history_messages_key="history",
        )

    def _get_tools(self):
        """统一管理工具列表"""
        return [
            rag_summarize, get_user_location, get_user_id,
            get_current_month, fetch_external_data, fill_context_for_report, mcp_weather_tool,mcp_bailian_web_search_tool
        ]

    def _wrap_agent_with_monitor(self, agent: Runnable) -> Runnable:
        """包装Agent，实现工具监控+动态提示词切换"""
        # 1. 模型调用前日志
        def log_before_model_wrapper(inputs: Dict[str, Any]) -> Dict[str, Any]:
            logger.info(f"[log_before_model] 用户输入：{inputs.get('input', '')}")
            logger.info(f"[log_before_model] 会话历史：{inputs.get('history', [])}")
            return inputs

        # 2. 工具调用监控（核心：标记报告场景）
        def wrap_tool(tool):
            original_func = tool.func

            def wrapped_func(*args, **kwargs):
                logger.info(f"[tool monitor] 执行工具：{tool.name}")
                logger.info(f"[tool monitor] 传入参数：args={args}, kwargs={kwargs}")
                try:
                    result = original_func(*args, **kwargs)
                    logger.info(f"[tool monitor] 工具{tool.name}调用成功，结果：{result}")

                    # 核心：调用fill_context_for_report时，标记当前会话为报告场景
                    if tool.name == "fill_context_for_report":
                        # 从kwargs/args中提取session_id（config里的configurable.session_id）
                        session_id = None
                        # 适配不同的参数传递方式
                        for arg in args:
                            if isinstance(arg, dict) and "configurable" in arg and "session_id" in arg["configurable"]:
                                session_id = arg["configurable"]["session_id"]
                                break
                        if not session_id and "config" in kwargs:
                            session_id = kwargs["config"].get("configurable", {}).get("session_id")

                        if session_id:
                            set_session_report_flag(session_id, True)
                            logger.info(f"[tool monitor] 会话{session_id}标记为报告生成场景")

                    return result
                except Exception as e:
                    logger.error(f"[tool monitor] 工具{tool.name}调用失败，原因：{str(e)}")
                    raise e

            tool.func = wrapped_func
            return tool

        # 包装所有工具
        for tool in self._get_tools():
            wrap_tool(tool)

        # 3. 动态提示词切换（核心：根据会话标记替换system_prompt）
        def dynamic_prompt_wrapper(inputs: Dict[str, Any]) -> Dict[str, Any]:
            # 提取session_id（从config中获取）
            session_id = inputs.get("config", {}).get("configurable", {}).get("session_id")
            if not session_id:
                # 兼容RunnableWithMessageHistory的参数格式
                session_id = inputs.get("configurable", {}).get("session_id")

            # 读取会话的报告标记
            is_report = get_session_report_flag(session_id) if session_id else False
            logger.info(f"[report_prompt_switch] 会话{session_id}报告场景标记：{is_report}")

            # 动态替换system_prompt
            inputs["system_prompt"] = load_report_prompts() if is_report else load_system_prompts()
            logger.info(f"[report_prompt_switch] 会话{session_id}使用提示词：{'报告专用' if is_report else '通用'}")

            return inputs

        # 组合监控逻辑和原agent
        return RunnableLambda(log_before_model_wrapper) | RunnableLambda(dynamic_prompt_wrapper) | agent

    def _custom_error_handler(self, error) -> str:
        """自定义解析失败处理"""
        logger.error(f"[解析错误] {str(error)}")
        if hasattr(error, 'llm_output') and error.llm_output:
            llm_answer = error.llm_output.strip()
            if "Final Answer:" in llm_answer:
                return llm_answer.split("Final Answer:")[-1].strip()
            else:
                return llm_answer
        return "我不知道"

    def execute_stream(self, query: str, config: dict):
        """流式执行（新增：执行前后处理会话标记）"""
        session_id = config.get("configurable", {}).get("session_id")
        # 执行前清空旧标记
        clear_session_report_flag(session_id)

        input_dict = {"input": query, "config": config}  # 传递config给动态提示词逻辑
        for chunk in self.agent.stream(input_dict, stream_mode="values", config=config):
            if "output" in chunk and chunk["output"]:
                yield chunk["output"].strip() + "\n"

        # 执行后清空标记
        clear_session_report_flag(session_id)

    def execute_invoke(self, query: str, config: dict):
        """同步执行（新增：执行前后处理会话标记）"""
        session_id = config.get("configurable", {}).get("session_id")
        # 执行前清空旧标记
        clear_session_report_flag(session_id)

        input_dict = {"input": query, "config": config}  # 传递config给动态提示词逻辑
        res = self.agent.invoke(input_dict, config=config)

        final_answer = res.get("output", "").strip()
        logger.info(f"[执行结果] 中间步骤：{res.get('intermediate_steps', [])}")
        if final_answer in ["Agent stopped due to iteration limit or time limit.", ""]:
            if "intermediate_steps" in res and len(res["intermediate_steps"]) > 0:
                last_step = res["intermediate_steps"][-1]
                if hasattr(last_step, 'observation'):
                    final_answer = last_step.observation
                else:
                    final_answer = "我不知道"
            else:
                final_answer = "我不知道"

        logger.info(f"[执行结果] 最终回答：{final_answer}")
        # 执行后清空标记
        clear_session_report_flag(session_id)
        return final_answer


if __name__ == '__main__':
    agent = ReactAgent()

    config = {
        "configurable": {
            "session_id": "user001",
        }
    }

    # # 测试1：普通工具调用（临猗天气，用通用提示词）
    # print("===== 测试普通工具调用 =====")
    # res = agent.execute_invoke("临猗今天的天气如何？", config=config)
    # print("最终回答：", res)

    # 测试2：报告生成场景（触发fill_context_for_report，自动切换为报告提示词）
    # print("\n===== 测试报告生成场景 =====")
    # res2 = agent.execute_invoke("生成我的2025-02使用报告", config=config)
    # print("最终回答：", res2)

    # 测试3：联网搜索能力
    print("\n===== 联网搜索场景 =====")
    res2 = agent.execute_invoke("我上一个问题是啥？", config=config)
    print("最终回答：", res2)


    # 测试流式调用（可选）
    # print("\n===== 测试流式输出 =====")
    # for chunk in agent.execute_stream("生成我的2026-02使用报告", config=config):
    #     print(chunk, end="", flush=True)