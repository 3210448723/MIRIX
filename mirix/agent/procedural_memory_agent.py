"""程序性记忆代理 (ProceduralMemoryAgent)

程序性记忆强调“如何做”与可重复执行的步骤序列。

典型内容：
    - 操作工作流（如：部署步骤、清洗数据流程）
    - 用户常用任务的参数偏好总结
    - 工具调用的成功模式（最佳参数组合）

潜在能力：
    1. 自动从多轮函数调用轨迹提炼出稳固“技能模版”
    2. 跟踪成功率 / 耗时，进行策略优化
    3. 与工具路由决策结合，优先推荐高胜率链路

当前：仅占位。
"""

from mirix.agent import Agent


class ProceduralMemoryAgent(Agent):
    def __init__(self, **kwargs):
        # 复用基类初始化
        super().__init__(**kwargs)