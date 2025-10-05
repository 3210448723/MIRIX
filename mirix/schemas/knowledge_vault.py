"""knowledge_vault.py
知识库 / 机密信息（Knowledge Vault）Schema。

用于存储敏感或高价值的持久化条目（如凭证、关键配置、书签等）。
仅添加中文注释，不修改字段和逻辑。
"""

from datetime import datetime
from typing import Dict, Optional, Any, List

from pydantic import Field, field_validator
from mirix.constants import MAX_EMBEDDING_DIM

from mirix.schemas.mirix_base import MirixBase
from mirix.schemas.embedding_config import EmbeddingConfig
from mirix.utils import get_utc_time


class KnowledgeVaultItemBase(MirixBase):
    """Base schema for knowledge vault items containing common fields.

    中文：知识库基础条目模型。包含类型（entry_type）、来源（source）、敏感级别（sensitivity）、核心值（secret_value）及说明（caption）。
    注意：这里并未直接区分是否需要加密存储，实际加密/脱敏可在数据访问层或存储层实现。
    """
    __id_prefix__ = "kv_item"
    entry_type: str = Field(..., description="Category (e.g., 'credential', 'bookmark', 'api_key')")
    source: str = Field(..., description="Information on who/where it was provided")
    sensitivity: str = Field(..., description="Data sensitivity level ('low', 'medium', 'high')")
    secret_value: str = Field(..., description="The actual credential or data value")
    caption: str = Field(..., description="Description of the knowledge vault item (e.g. 'API key for OpenAI Service')")


class KnowledgeVaultItem(KnowledgeVaultItemBase):
    """Representation of a knowledge vault item for storing credentials, bookmarks, etc.

    中文：持久化实体，包含 id / 创建 / 更新时间，last_modify 用于记录最近操作。
    caption_embedding：针对 caption 的向量，用于语义检索（如查找某类凭证说明）。
    metadata_：可存储标签、权限标识、使用统计等。
    """
    id: Optional[str] = Field(None, description="Unique identifier for the knowledge vault item")
    user_id: str = Field(..., description="The id of the user who generated the knowledge vault item")
    created_at: datetime = Field(default_factory=get_utc_time, description="The creation date of the knowledge vault item")
    updated_at: Optional[datetime] = Field(None, description="The last update date of the knowledge vault item")
    last_modify: Dict[str, Any] = Field(
        default_factory=lambda: {"timestamp": get_utc_time().isoformat(), "operation": "created"},
        description="Last modification info including timestamp and operation type"
    )
    metadata_: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary additional metadata")
    organization_id: str = Field(..., description="The unique identifier of the organization")
    caption_embedding: Optional[List[float]] = Field(None, description="The embedding of the summary")
    embedding_config: Optional[EmbeddingConfig] = Field(None, description="The embedding configuration used by the event")

    # need to validate both details_embedding and summary_embedding to ensure they are the same size
    @field_validator("caption_embedding")
    @classmethod
    def pad_embeddings(cls, embedding: List[float]) -> List[float]:
        """Pad embeddings to `MAX_EMBEDDING_SIZE`. This is necessary to ensure all stored embeddings are the same size."""
        import numpy as np

        if embedding and len(embedding) != MAX_EMBEDDING_DIM:
            np_embedding = np.array(embedding)
            padded_embedding = np.pad(np_embedding, (0, MAX_EMBEDDING_DIM - np_embedding.shape[0]), mode="constant")
            return padded_embedding.tolist()
        return embedding

class KnowledgeVaultItemCreate(KnowledgeVaultItemBase):
    """Schema for creating a new knowledge vault item.

    中文：创建模型，直接复用基础字段；若需要额外安全校验由上层逻辑处理。
    """
    pass


class KnowledgeVaultItemUpdate(MirixBase):
    """Schema for updating an existing knowledge vault item.

    中文：Patch 更新模型。除 id 外全部可选，提供即修改。updated_at 自动刷新。
    可用于轮换 secret_value、修改说明 caption、调整敏感级别或添加 metadata。
    """
    id: str = Field(..., description="Unique ID for this knowledge vault entry")
    entry_type: Optional[str] = Field(None, description="Category (e.g., 'credential', 'bookmark', 'api_key')")
    source: Optional[str] = Field(None, description="Information on who/where it was provided")
    sensitivity: Optional[str] = Field(None, description="Data sensitivity level ('low', 'medium', 'high')")
    secret_value: Optional[str] = Field(None, description="The actual credential or data value")
    metadata_: Optional[Dict[str, Any]] = Field(None, description="Arbitrary additional metadata")
    organization_id: Optional[str] = Field(None, description="The unique identifier of the organization")
    updated_at: datetime = Field(default_factory=get_utc_time, description="The update date")
    last_modify: Optional[Dict[str, Any]] = Field(
        None,
        description="Last modification info including timestamp and operation type"
    )
    caption_embedding: Optional[List[float]] = Field(None, description="The embedding of the summary")
    embedding_config: Optional[EmbeddingConfig] = Field(None, description="The embedding configuration used by the event")

class KnowledgeVaultItemResponse(KnowledgeVaultItem):
    """Response schema for knowledge vault items with additional fields that might be needed by the API.

    中文：响应模型（当前未扩展，可作为未来附加脱敏处理或权限过滤的输出层）。
    """
    pass
