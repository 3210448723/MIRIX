"""知识库保险库记忆代理 (KnowledgeVaultAgent)

定位：权威、精选、结构化、高信噪比的知识集合（可人工审核）。

与语义记忆差异：
        - 更强调条目治理：版本 / 来源 / 可信度 / 审核状态
        - 支持分层命名空间、引用追踪
        - 适合作为“模型回答引用”的来源集（减少幻觉）

规划扩展：
        1. CRDT / 版本合并策略（多人编辑）
        2. 引用计数驱动的淘汰/压缩
        3. 与内部 Wiki / 外部文档同步
        4. 条目差异审阅（diff + 摘要）

当前：仅继承 `Agent`；无附加逻辑。
"""

from mirix.agent import Agent


class KnowledgeVaultAgent(Agent):
    def __init__(self, **kwargs):
        # 复用父类初始化；外部会传入统一的 llm_config、状态追踪等 kwargs
        super().__init__(**kwargs)