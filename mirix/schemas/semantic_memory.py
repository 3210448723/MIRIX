"""semantic_memory.py
语义记忆（Semantic Memory）数据模型。

语义记忆用于存储抽象、概念性、相对稳定的知识点（区别于 episodic 的具体时间事件）。
结构划分：
1. SemanticMemoryItemBase：公共内容字段。
2. SemanticMemoryItem：实际持久化实体，含 id / 时间戳 / 嵌入向量 / 组织与用户上下文。
3. SemanticMemoryItemUpdate：Patch 模型，仅更新提供的字段。

嵌入说明：summary / details / name 三段文本可分别生成向量，用于语义检索。
验证器会自动做零填充，确保长度统一到 MAX_EMBEDDING_DIM，便于数据库列定长存储或向量索引。

仅添加中文注释，不改变任何业务逻辑。
"""

from datetime import datetime
from typing import Dict, List, Optional, Any

from pydantic import Field, field_validator
from mirix.constants import MAX_EMBEDDING_DIM

from mirix.schemas.mirix_base import MirixBase
from mirix.utils import get_utc_time
from mirix.schemas.embedding_config import EmbeddingConfig

class SemanticMemoryItemBase(MirixBase):
    """Base schema for storing semantic memory items (e.g., general knowledge, concepts, facts).

    中文：语义记忆基础模型。
    name/summary/details：表示知识点的名称、摘要与详细描述。
    source：信息来源（便于溯源 / 可信度评估）。
    tree_path：层级分类路径（如 ['favorites','pets','dog']），支持分层过滤与聚合。
    """
    __id_prefix__ = "sem_item"
    name: str = Field(..., description="The name or main concept/object for the knowledge entry")
    summary: str = Field(..., description="A concise explanation or summary of the concept")
    details: str = Field(..., description="Detailed explanation or additional context for the concept")
    source: str = Field(..., description="Reference or origin of this information (e.g., book, article, movie)")
    tree_path: List[str] = Field(..., description="Hierarchical categorization path as an array of strings (e.g., ['favorites', 'pets', 'dog'])")

class SemanticMemoryItem(SemanticMemoryItemBase):
    """Full semantic memory item schema, including database-related fields.

    中文：语义记忆实体，含持久化必需的 id / created_at / updated_at / last_modify。
    * embeddings: name / summary / details 各自可生成独立向量；缺失时为 None。
    * last_modify：记录最近操作（operation + timestamp），便于审计或排序。
    * 统一维度填充在 pad_embeddings 验证器中完成。
    """
    id: Optional[str] = Field(None, description="Unique identifier for the semantic memory item")
    user_id: str = Field(..., description="The id of the user who generated the semantic memory")
    created_at: datetime = Field(default_factory=get_utc_time, description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")
    last_modify: Dict[str, Any] = Field(
        default_factory=lambda: {"timestamp": get_utc_time().isoformat(), "operation": "created"},
        description="Last modification info including timestamp and operation type"
    )
    metadata_: Dict[str, Any] = Field(default_factory=dict, description="Additional arbitrary metadata as a JSON object")
    organization_id: str = Field(..., description="The unique identifier of the organization")
    details_embedding: Optional[List[float]] = Field(None, description="The embedding of the details")
    name_embedding: Optional[List[float]] = Field(None, description="The embedding of the name")
    summary_embedding: Optional[List[float]] = Field(None, description="The embedding of the summary")
    embedding_config: Optional[EmbeddingConfig] = Field(None, description="The embedding configuration used by the event")

    # need to validate both details_embedding and summary_embedding to ensure they are the same size
    @field_validator("details_embedding", "summary_embedding", "name_embedding")
    @classmethod
    def pad_embeddings(cls, embedding: List[float]) -> List[float]:
        """Pad embeddings to `MAX_EMBEDDING_SIZE`. This is necessary to ensure all stored embeddings are the same size."""
        import numpy as np

        if embedding and len(embedding) != MAX_EMBEDDING_DIM:
            np_embedding = np.array(embedding)
            padded_embedding = np.pad(np_embedding, (0, MAX_EMBEDDING_DIM - np_embedding.shape[0]), mode="constant")
            return padded_embedding.tolist()
        return embedding

class SemanticMemoryItemUpdate(MirixBase):
    """Schema for updating an existing semantic memory item.

    中文：语义记忆 Patch 更新模型。除 id 外字段均可选；仅对提供内容执行局部更新。
    updated_at 自动刷新；上层逻辑可在检测到文本字段变化时重新生成对应 embedding。
    """
    id: str = Field(..., description="Unique ID for this semantic memory entry")
    name: Optional[str] = Field(None, description="The name or main concept for the knowledge entry")
    summary: Optional[str] = Field(None, description="A concise explanation or summary of the concept")
    details: Optional[str] = Field(None, description="Detailed explanation or additional context for the concept")
    source: Optional[str] = Field(None, description="Reference or origin of this information (e.g., book, article, movie)")
    actor: Optional[str] = Field(None, description="The actor who generated the semantic memory (user or assistant)")
    tree_path: Optional[List[str]] = Field(
        None, 
        description="Hierarchical categorization path as an array of strings (e.g., ['favorites', 'pets', 'dog'])"
    )
    metadata_: Optional[Dict[str, Any]] = Field(None, description="Additional arbitrary metadata as a JSON object")
    organization_id: Optional[str] = Field(None, description="The organization ID")
    updated_at: datetime = Field(default_factory=get_utc_time, description="Update timestamp")
    last_modify: Optional[Dict[str, Any]] = Field(
        None,
        description="Last modification info including timestamp and operation type"
    )
    details_embedding: Optional[List[float]] = Field(None, description="The embedding of the details")
    name_embedding: Optional[List[float]] = Field(None, description="The embedding of the name")
    summary_embedding: Optional[List[float]] = Field(None, description="The embedding of the summary")
    embedding_config: Optional[EmbeddingConfig] = Field(None, description="The embedding configuration used by the event")


class SemanticMemoryItemResponse(SemanticMemoryItem):
    """
    Response schema for semantic memory item.
    """
    pass
