class AgentStates:
    """Agent 状态聚合容器

    作用：集中保存系统内多个逻辑 Agent 的运行状态引用，便于：
        - 统一注入到消息路由 / 记忆分发
        - 动态扩展（添加新类型 Memory Agent）

    字段说明：
        agent_state: 主对话 / Orchestrator 代理状态
        episodic_memory_agent_state: 情节记忆代理
        procedural_memory_agent_state: 程序性记忆代理
        knowledge_vault_agent_state: 知识金库代理
        meta_memory_agent_state: 元记忆（决策 / 统计）
        semantic_memory_agent_state: 语义记忆代理
        core_memory_agent_state: 核心长期记忆代理
        resource_memory_agent_state: 资源引用记忆代理
        reflexion_agent_state: 反思 / 自评估代理（若实现）
        background_agent_state: 后台监控 / 周期任务代理
    """
    
    def __init__(self):
        self.agent_state = None
        self.episodic_memory_agent_state = None
        self.procedural_memory_agent_state = None
        self.knowledge_vault_agent_state = None
        self.meta_memory_agent_state = None
        self.semantic_memory_agent_state = None
        self.core_memory_agent_state = None
        self.resource_memory_agent_state = None
        self.reflexion_agent_state = None
        self.background_agent_state = None
        
    def set_agent_state(self, name, state):
        """按属性名设置对应 Agent 状态。

        参数：
            name: 属性字符串（需与类字段同名）
            state: 具体 AgentState 实例
        异常：
            ValueError: 若 name 不在已定义字段中。
        """
        if hasattr(self, name):
            setattr(self, name, state)
        else:
            raise ValueError(f"Unknown agent state name: {name}")
    
    def get_agent_state(self, name):
        """按名称获取 Agent 状态。

        用于动态路由：例如根据字符串 agent_type 取对应状态。
        """
        if hasattr(self, name):
            return getattr(self, name)
        else:
            raise ValueError(f"Unknown agent state name: {name}")
    
    def get_all_states(self):
        """以字典形式返回所有已注册状态（便于序列化 / 调试）。"""
        return {
            'agent_state': self.agent_state,
            'episodic_memory_agent_state': self.episodic_memory_agent_state,
            'procedural_memory_agent_state': self.procedural_memory_agent_state,
            'knowledge_vault_agent_state': self.knowledge_vault_agent_state,
            'meta_memory_agent_state': self.meta_memory_agent_state,
            'semantic_memory_agent_state': self.semantic_memory_agent_state,
            'core_memory_agent_state': self.core_memory_agent_state,
            'resource_memory_agent_state': self.resource_memory_agent_state,
        } 