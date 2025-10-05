"""语义记忆代理 (SemanticMemoryAgent)

语义记忆用于存放“可复用的事实/概念性知识”，不强依赖具体时间点。

与其他记忆对比：
    - 情节记忆：时间序列事件 -> 语义记忆：去时间化的事实表达
    - 程序性记忆：操作步骤/技能
    - 知识金库：更权威 + 结构化 + 审核流程

未来扩展：
    1. Embedding 向量检索（相似度 + 关键词混合检索）
    2. 事实冲突检测（来源权重 + 最新优先）
    3. 自动归纳：从情节记忆批量聚合出稳定模式写入
    4. 与外部文档数据库同步（定期 ETL）

当前：仅继承 `Agent`。
"""

from mirix.agent import Agent


class SemanticMemoryAgent(Agent):
    def __init__(self, **kwargs):
        # 调用基础 Agent 初始化
        super().__init__(**kwargs)