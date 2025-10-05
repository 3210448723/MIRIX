"""核心记忆代理 (CoreMemoryAgent)

职责概述：
    管理系统中“核心 / 长期”事实性记忆，代表极稳定、跨任务复用的信息。

典型内容：
    - 用户的基础档案与长期偏好（若不放在用户 Profile 中）
    - 长期不易过期的关键业务规则摘要
    - 跨多会话持续引用的身份/目标设定

未来扩展方向（尚未实现）：
    1. 事实去重与冲突消解（按哈希 / 语义相似度阈值合并）
    2. 可信度加权与来源追踪（source + weight）
    3. 与语义记忆、知识金库建立交叉引用（ID 链接）
    4. 定期压缩低访问频次条目（归档或降权）

当前实现：
    仅继承 `Agent`，不额外覆写方法，作为扩展挂载点。
"""

from mirix.agent import Agent


class CoreMemoryAgent(Agent):
    def __init__(self, **kwargs):
        # 调用父类初始化：传入的 **kwargs 由外部构造统一控制
        super().__init__(**kwargs)