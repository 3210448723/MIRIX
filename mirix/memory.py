"""memory.py
记忆（Memory）相关的通用工具函数。

本文件目前主要包含：
1. get_memory_functions: 动态收集 Memory 子类中可暴露的函数，供函数调用或工具化使用。
2. summarize_messages: 使用 LLM 对一段消息历史进行总结，减少上下文占用。

注释原则：
 - 不修改任何执行逻辑，只添加中文说明。
 - 若某些英文常量/提示词对模型行为有影响，不翻译其内容，只在旁边加中文注释。
 - 递归截断（truncation）策略保留原貌，仅解释其目的。
"""

from typing import Callable, Dict, List, Optional

from mirix.constants import MESSAGE_SUMMARY_REQUEST_ACK  # 用于与模型建立“总结请求已确认”的对话上下文的固定回复（不要改动其英文内容，英文原样确保提示链稳定）
from mirix.llm_api.llm_api_tools import create  # 目前未直接使用（保留引用以兼容历史/未来调用）
from mirix.prompts.gpt_summarize import SYSTEM as SUMMARY_PROMPT_SYSTEM  # 系统级总结提示词（英文内容本身勿改）
from mirix.schemas.agent import AgentState
from mirix.schemas.enums import MessageRole
from mirix.schemas.memory import Memory
from mirix.schemas.message import Message
from mirix.schemas.mirix_message_content import TextContent
from mirix.settings import summarizer_settings  # 总结器相关配置（阈值等）
from mirix.utils import count_tokens, printd
from mirix.llm_api.llm_client import LLMClient

def get_memory_functions(cls: Memory) -> Dict[str, Callable]:
    """收集 Memory 子类中的可调用函数。

    设计目的：
    - 用于 *动态* 枚举某个 Memory 实现里可暴露给外部（比如函数调用 / 工具集成）的函数。
    - 排除基类 `Memory` 中定义的通用/内部函数，避免污染工具列表。

    过滤规则：
    1. 跳过以下情况：
       - 名称以下划线开头（视为私有/内部）。
       - 显式在排除列表: ['load', 'to_dict']。
       - 出现在基类 `Memory` 中（避免重复/继承方法被暴露）。
    2. 仅保留可调用对象（callable）。

    参数:
        cls: Memory 的具体子类。

    返回:
        一个 {函数名: 函数对象} 的字典，用于后续注册或调用。
    """
    functions: Dict[str, Callable] = {}

    # 收集基类 Memory 中的所有可调用成员，后续用来排除
    base_functions: List[str] = []  # 基类 Memory 的全部方法名，用于排除
    for func_name in dir(Memory):
        funct = getattr(Memory, func_name)
        if callable(funct):
            base_functions.append(func_name)

    # 遍历子类自身的属性
    for func_name in dir(cls):
        # 1) 跳过私有或显式排除的方法
        if func_name.startswith("_") or func_name in ["load", "to_dict"]:  # 这些通常是内部或序列化方法，不向外暴露
            continue
        # 2) 若函数名属于基类方法，则忽略（不暴露基类通用方法）
        if func_name in base_functions:
            continue
        func = getattr(cls, func_name)
        # 3) 必须是可调用对象
        if not callable(func):
            continue
        functions[func_name] = func
    return functions


def _format_summary_history(message_history: List[Message]):
    """将消息历史格式化为用于总结的纯文本。

    说明：
    - 当前实现是一个简单拼接器：role + 内容文本。
    - TODO 注释提示未来可替换为更标准的格式（例如 ChatML）。
    - 多模态内容（图片/文件/云端文件）以占位符方式标记，便于模型理解上下文类型。
    """
    # TODO: 未来可复用统一的 prompt 格式化工具（例如 ChatML）
    def format_message(m: Message):
        content_str = ''
        for content in m.content:
            if content.type == 'text':  # 纯文本内容
                content_str += content.text + "\n"
            elif content.type == 'image_url':  # 图片占位符
                content_str += f"[Image: {content.image_id}]" + "\n"
            elif content.type == 'file_uri':  # 本地/存储文件占位符
                content_str += f"[File: {content.file_id}]" + "\n"
            elif content.type == 'google_cloud_file_uri':  # 云端文件占位符
                content_str += f"[Cloud File: {content.cloud_file_uri}]" + "\n"
            else:  # 未知类型占位符（保留以便调试）
                content_str += f"[Unknown content type: {content.type}]" + "\n"
        return content_str.strip()
    # 用两个换行分隔不同消息，保持清晰结构
    return "\n\n".join([f"{m.role}: {format_message(m)}" for m in message_history])


def summarize_messages(
    agent_state: AgentState,
    message_sequence_to_summarize: List[Message],
    existing_file_uris: Optional[List[str]] = None,
):
    """利用 LLM 对消息序列进行总结（压缩上下文）。

    功能概述：
    1. 将一段对话历史格式化后送入总结提示词，让模型生成总结文本。
    2. 若原始文本 token 数过大，采用递归截断策略，逐步减少输入规模。
    3. 使用与对话不同的 system + assistant ACK + user 输入的对话结构，以帮助模型理解“当前是一个总结任务”。

    关键点：
    - `SUMMARY_PROMPT_SYSTEM`：系统提示词，指定“你要做总结”这一角色语境（内容不要改）。
    - `MESSAGE_SUMMARY_REQUEST_ACK`：模拟助手已收到“请做总结”的确认（保持英文，防止 prompt 流程失效）。
    - 递归策略：当文本超过阈值（memory_warning_threshold * context_window）时，截取前段进行一次总结，然后把返回的总结结果与剩余尾部拼接，再继续总结，从而降低 token 占用。

    参数：
        agent_state: 当前 Agent 的状态对象（包含 LLM 配置等）。
        message_sequence_to_summarize: 需要被总结的消息列表（按时间顺序）。
        existing_file_uris: （可选）已有的文件 URI，传递给底层 LLM 以便引用上下文文件。

    返回：
        模型生成的总结文本（字符串）。
    """
    # 读取上下文窗口大小，用于计算截断阈值
    context_window = agent_state.llm_config.context_window

    summary_prompt = SUMMARY_PROMPT_SYSTEM  # 系统提示词（保持英文原文）
    summary_input = _format_summary_history(message_sequence_to_summarize)
    summary_input_tkns = count_tokens(summary_input)

    # 当输入 token 数超过阈值时触发分段递归总结。
    if summary_input_tkns > summarizer_settings.memory_warning_threshold * context_window:
        # trunc_ratio：根据阈值计算需要保留的比例，再乘 0.8 作为安全裕量，避免刚好触顶。
        trunc_ratio = (summarizer_settings.memory_warning_threshold * context_window / summary_input_tkns) * 0.8  # 安全系数 0.8
        cutoff = int(len(message_sequence_to_summarize) * trunc_ratio)
        # 递归调用：
        #   1) 先对前半段（较早消息）做一次 summarize_messages -> 得到一个“阶段总结”作为列表第 0 元素
        #   2) 然后追加后半段原始消息
        #   3) 将其转为字符串再作为新的输入，这样逐步缩短整体长度
        summary_input = str(
            [
                summarize_messages(
                    agent_state,
                    message_sequence_to_summarize=message_sequence_to_summarize[:cutoff]
                )
            ] + message_sequence_to_summarize[cutoff:]
        )

    dummy_agent_id = agent_state.id  # 用 Agent 自身 id 伪造一段“总结任务”对话上下文
    message_sequence = [
        # system: 设定总结任务的角色/目标
        Message(agent_id=dummy_agent_id, role=MessageRole.system, content=[TextContent(text=summary_prompt)]),
        # assistant: 发送一个确认 ACK，帮助模型进入“准备总结”状态
        Message(agent_id=dummy_agent_id, role=MessageRole.assistant, content=[TextContent(text=MESSAGE_SUMMARY_REQUEST_ACK)]),
        # user: 以用户身份提供需要被总结的原始消息串
        Message(agent_id=dummy_agent_id, role=MessageRole.user, content=[TextContent(text=summary_input)]),
    ]

    # TODO: 未来可为 summarizer 使用专门的、更便宜或更适配的 LLM 配置
    llm_config_no_inner_thoughts = agent_state.llm_config.model_copy(deep=True)
    llm_config_no_inner_thoughts.put_inner_thoughts_in_kwargs = False  # 禁用内部思考注入，减少污染和 token 浪费

    # 早期可能使用过 create(...)，此处改为统一的 LLMClient 封装
    llm_client = LLMClient.create(
        llm_config=llm_config_no_inner_thoughts,
    )
    response = llm_client.send_llm_request(
        messages=message_sequence,
        existing_file_uris=existing_file_uris,
    )

    printd(f"summarize_messages gpt reply: {response.choices[0]}")  # 调试输出（可在生产中根据日志级别控制）
    reply = response.choices[0].message.content
    return reply
