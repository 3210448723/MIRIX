"""schemas/memory.py
内存（Memory）相关的数据结构定义。

核心目标：
1. 用统一的 Block 列表描述“核心记忆”（Core / In-Context Memory）结构。
2. 提供 Jinja2 模板，将多个记忆块渲染为可注入 LLM system prompt 的字符串。
3. 定义与上下文窗口（Context Window）分析相关的统计 Schema，方便调试 / 可视化。
4. 给不同类型的 memory（如 ChatMemory）提供基础实现与扩展入口。

注释策略：
 - 不改变任何字段名称与逻辑。
 - 引用的英文提示 / 结构保持原样，只在旁边补充中文说明。
 - 对潜在的改进点（TODO）添加中文解释，方便后续维护。
"""

from typing import TYPE_CHECKING, List, Optional

from jinja2 import Template, TemplateSyntaxError, Environment
from pydantic import BaseModel, Field

# Forward referencing to avoid circular import with Agent -> Memory -> Agent
if TYPE_CHECKING:
    pass

from mirix.constants import CORE_MEMORY_BLOCK_CHAR_LIMIT
from mirix.schemas.block import Block
from mirix.schemas.message import Message
from mirix.schemas.openai.chat_completion_request import Tool
from mirix.schemas.user import User as PydanticUser

class ContextWindowOverview(BaseModel):
    """上下文窗口（Context Window）使用情况概览。

    该结构用于在调试或分析时展示：
    - 当前加载进 LLM 上下文的 token 占用分布
    - 系统提示词 / 核心记忆 / 函数定义 / 消息 列表所占 tokens
    - 外部记忆（归档 + 回忆）汇总占用
    有助于分析截断策略是否合理、提示组合是否过长。
    """

    # top-level information
    context_window_size_max: int = Field(..., description="The maximum amount of tokens the context window can hold.")
    context_window_size_current: int = Field(..., description="The current number of tokens in the context window.")

    # context window breakdown (in messages)
    # (technically not in the context window, but useful to know)
    num_messages: int = Field(..., description="The number of messages in the context window.")
    num_archival_memory: int = Field(..., description="The number of messages in the archival memory.")
    num_recall_memory: int = Field(..., description="The number of messages in the recall memory.")
    num_tokens_external_memory_summary: int = Field(
        ..., description="The number of tokens in the external memory summary (archival + recall metadata)."
    )
    external_memory_summary: str = Field(
        ..., description="The metadata summary of the external memory sources (archival + recall metadata)."
    )

    # context window breakdown (in tokens)
    # this should all add up to context_window_size_current

    num_tokens_system: int = Field(..., description="The number of tokens in the system prompt.")
    system_prompt: str = Field(..., description="The content of the system prompt.")

    num_tokens_core_memory: int = Field(..., description="The number of tokens in the core memory.")
    core_memory: str = Field(..., description="The content of the core memory.")

    num_tokens_summary_memory: int = Field(..., description="The number of tokens in the summary memory.")
    summary_memory: Optional[str] = Field(None, description="The content of the summary memory.")

    num_tokens_functions_definitions: int = Field(..., description="The number of tokens in the functions definitions.")
    functions_definitions: Optional[List[Tool]] = Field(..., description="The content of the functions definitions.")

    num_tokens_messages: int = Field(..., description="The number of tokens in the messages list.")
    # TODO make list of messages?
    # messages: List[dict] = Field(..., description="The messages in the context window.")
    messages: List[Message] = Field(..., description="The messages in the context window.")

def line_numbers(value: str, prefix: str = "Line ") -> str:
    """
    Turn  
        "a\nb"  
    into  
        "Line 1:\ta\nLine 2:\tb"
    """
    return "\n".join(
        f"{prefix}{idx + 1}:\t{line}"
        for idx, line in enumerate(value.splitlines())
    )

# Build an environment and add a custom filter
env = Environment()
env.filters["line_numbers"] = line_numbers

class Memory(BaseModel, validate_assignment=True):
    """表示 Agent 的“上下文内记忆”（In-Context / Core Memory）。

    设计说明：
    - 通过一组 `Block`（带 label / limit / value）来组织多段结构化信息，如 persona / human 设定。
    - 支持使用 Jinja2 模板自定义最终注入到 LLM 的拼接格式（`compile()`）。
    - 未来可扩展：Block 级别的权限控制、动态裁剪、权重排序等。
    """

    # Memory.block contains the list of memory blocks in the core memory
    blocks: List[Block] = Field(..., description="Memory blocks contained in the agent's in-context memory")

    # Memory.template is a Jinja2 template for compiling memory module into a prompt string.
    prompt_template: str = Field(
        default="{% for block in blocks %}"
        '<{{ block.label }} characters="{{ block.value|length }}/{{ block.limit }}">\n'
        "{{ block.value|line_numbers }}\n"
        "</{{ block.label }}>"
        "{% if not loop.last %}\n{% endif %}"
        "{% endfor %}",
        description="Jinja2 template for compiling memory blocks into a prompt string",
    )

    def get_prompt_template(self) -> str:
        """返回当前使用的 Jinja2 模板字符串。"""
        return str(self.prompt_template)

    def set_prompt_template(self, prompt_template: str):
        """设置新的 Jinja2 模板。

        校验步骤：
        1. 语法是否有效（Template 解析）。
        2. 使用当前 blocks 进行一次渲染测试，验证上下文变量可用。
        若任一步骤失败，抛出详细错误，方便上层处理。
        """
        try:
            # Validate Jinja2 syntax
            Template(prompt_template)

            # Validate compatibility with current memory structure
            Template(prompt_template).render(blocks=self.blocks)

            # If we get here, the template is valid and compatible
            self.prompt_template = prompt_template
        except TemplateSyntaxError as e:
            raise ValueError(f"Invalid Jinja2 template syntax: {str(e)}")
        except Exception as e:
            raise ValueError(f"Prompt template is not compatible with current memory structure: {str(e)}")

    def compile(self) -> str:
        """根据当前 Jinja2 模板，将所有记忆块渲染为最终字符串。

        返回值通常会被拼接到 system prompt 或前置上下文中。
        """
        template = env.from_string(self.prompt_template)
        return template.render(blocks=self.blocks)

    def list_block_labels(self) -> List[str]:
        """返回所有记忆块的 label 列表。"""
        # return list(self.memory.keys())
        return [block.label for block in self.blocks]

    # TODO: these should actually be label, not name
    def get_block(self, label: str) -> Block:
        """按 label 获取对应 Block；若不存在则抛出 KeyError。"""
        keys = []
        for block in self.blocks:
            if block.label == label:
                return block
            keys.append(block.label)
        raise KeyError(f"Block field {label} does not exist (available sections = {', '.join(keys)})")

    def get_blocks(self) -> List[Block]:
        """返回当前持有的全部 Block 列表。"""
        # return list(self.memory.values())
        return self.blocks

    def set_block(self, block: Block):
        """插入或更新一个 Block：
        - 若 label 已存在则覆盖原值
        - 否则追加到列表末尾
        """
        for i, b in enumerate(self.blocks):
            if b.label == block.label:
                self.blocks[i] = block
                return
        self.blocks.append(block)

    def update_block_value(self, label: str, value: str):
        """更新指定 label 的 Block.value。

        限制：当前仅允许传入字符串，后续若支持结构化内容可调整类型校验。
        """
        if not isinstance(value, str):
            raise ValueError(f"Provided value must be a string")

        for block in self.blocks:
            if block.label == label:
                block.value = value
                return
        raise ValueError(f"Block with label {label} does not exist")


# TODO: ideally this is refactored into ChatMemory and the subclasses are given more specific names.
class BasicBlockMemory(Memory):
    """
    BasicBlockMemory is a basic implemention of the Memory class, which takes in a list of blocks and links them to the memory object. These are editable by the agent via the core memory functions.

    Attributes:
        memory (Dict[str, Block]): Mapping from memory block section to memory block.

    Methods:
        core_memory_append: Append to the contents of core memory.
        core_memory_rewrite: Rewrite the contents of core memory.
    """

    def __init__(self, blocks: List[Block] = []):
        """初始化 BasicBlockMemory。

        参数：
            blocks: 预置的 Block 列表，可为空。注意 list 作为默认值在可变语义上有副作用，
                    这里因使用方式简单且多数情况传入显式值，暂保持原状（可改为 None + 工厂）。
        """
        super().__init__(blocks=blocks)

    def core_memory_append(agent_state: "AgentState", label: str, content: str) -> Optional[str]:  # type: ignore
        """向核心记忆指定块追加文本内容（末尾换行后拼接）。

        参数：
            label: 目标 Block 的标签（如 persona / human）。
            content: 追加的纯文本（支持 Unicode & Emoji）。

        返回：始终返回 None（该函数本质是副作用更新）。
        """
        current_value = str(agent_state.memory.get_block(label).value)
        new_value = current_value + "\n" + str(content)
        agent_state.memory.update_block_value(label=label, value=new_value)
        return None

    # def core_memory_replace(agent_state: "AgentState", label: str, old_content: str, new_content: str) -> Optional[str]:  # type: ignore
    #     """
    #     Replace the contents of core memory. To delete memories, use an empty string for new_content.

    #     Args:
    #         label (str): Section of the memory to be edited (persona or human).
    #         old_content (str): String to replace. Must be an exact match.
    #         new_content (str): Content to write to the memory. All unicode (including emojis) are supported.

    #     Returns:
    #         Optional[str]: None is always returned as this function does not produce a response.
    #     """
    #     current_value = str(agent_state.memory.get_block(label).value)
    #     if old_content not in current_value:
    #         raise ValueError(f"Old content '{old_content}' not found in memory block '{label}'")
    #     new_value = current_value.replace(str(old_content), str(new_content))
    #     agent_state.memory.update_block_value(label=label, value=new_value)
    #     return None


class ChatMemory(BasicBlockMemory):
    """聊天场景专用核心记忆。

    默认初始化两个 Block：
    - persona: 角色/系统人格设定
    - human:   用户（人类）偏好/背景
    """

    def __init__(self, persona: str, human: str, actor: PydanticUser, limit: int = CORE_MEMORY_BLOCK_CHAR_LIMIT):
        """初始化 ChatMemory。

        参数：
            persona: 初始 persona 设定文本。
            human: 初始 human 设定文本。
            actor: 当前用户（用于标记 user_id）。
            limit: 每个 Block 的字符限制（来自常量 `CORE_MEMORY_BLOCK_CHAR_LIMIT`）。
        """
        # TODO: Should these be CreateBlocks?
        super().__init__(blocks=[Block(value=persona, limit=limit, label="persona", user_id=actor.id), Block(value=human, limit=limit, label="human", user_id=actor.id)])


class UpdateMemory(BaseModel):
    """更新 Memory 的指令结构占位符（当前未添加字段，后续可扩展）。"""


class ArchivalMemorySummary(BaseModel):
    """归档记忆汇总信息（数量级统计）。

    用途：在上下文窗口使用分析 / Telemetry 面板中快速展示归档池大小，
    便于判断是否需要触发进一步压缩或清理。
    """
    size: int = Field(..., description="归档记忆（archival memory）中的记录行数")


class RecallMemorySummary(BaseModel):
    """回忆记忆（近期高相关记忆）汇总信息。"""
    size: int = Field(..., description="回忆记忆（recall memory）中的记录行数")


class CreateArchivalMemory(BaseModel):
    """创建一条新的归档记忆输入结构。

    字段：
        text: 需要持久化的原始文本（尚未被总结或结构化）。
    """
    text: str = Field(..., description="写入归档记忆的原始文本内容")
