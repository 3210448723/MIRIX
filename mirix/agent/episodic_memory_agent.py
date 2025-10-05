"""情节记忆代理 (EpisodicMemoryAgent)

情节记忆用于记录“发生了什么”这一序列化事件流，强调时间上下文。

典型条目：
    - 用户最近进行的操作/命令
    - 关键对话轮次摘录
    - 任务阶段性里程碑（开始/暂停/完成）

设计目标：
    - 为行为推断 / 任务连续性 / 复盘提供依据
    - 可被总结器周期性压缩为更高层语义（写回语义记忆或核心记忆）

潜在扩展：
    - 时间衰减：对很久未引用的事件降低检索权重
    - Session segmentation：自动划分会话段落（聚类 / 主题漂移检测）
    - 与多模态上下文联动（截图 / 语音转写）形成富上下文时间线

当前：仅继承 `Agent` 占位。
"""

from mirix.agent import Agent


class EpisodicMemoryAgent(Agent):
    def __init__(self, **kwargs):
        # 复用父类初始化；参数由外部统一注入
        super().__init__(**kwargs)
