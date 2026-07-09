import torch
import torch.nn as nn


class ArkGuesserModelV2(nn.Module):
    """
    Optimized version of ArkGuesserModelV1.

    输入: x, shape (batch, 2, num_classes)
    输出: logits, shape (batch, 2)

    与 V1 的区别:
      1. 轻量级数量编码器: 单层 Linear(1, hidden_dim) + GELU 替代 3 层 MLP
      2. 残差交叉注意力 + LayerNorm: 保留原始信息 + 稳定训练
      3. 更小的 hidden_dim (128 vs 256), 更少的 heads (4 vs 8)
      4. 更低的 dropout (0.1 vs 0.25)
    """

    def __init__(
        self,
        num_classes: int,
        hidden_dim: int = 128,
        attn_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.n_heads = attn_heads

        # Class embedding
        self.class_emb = nn.Embedding(num_classes, hidden_dim)

        # Lightweight count encoder: scalar → hidden_dim (single layer, not deep MLP)
        self.count_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
        )

        # Cross-attention with residual connection + LayerNorm
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=attn_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_norm1 = nn.LayerNorm(hidden_dim)
        self.cross_norm2 = nn.LayerNorm(hidden_dim)

        # Learnable pooling query for PMA-style aggregation
        self.pool_seed = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.pool = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=attn_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.pool_norm1 = nn.LayerNorm(hidden_dim)
        self.pool_norm2 = nn.LayerNorm(hidden_dim)

        # Lightweight comparator: antisymmetric input
        self.comparator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.Embedding):
                nn.init.xavier_uniform_(module.weight)
            elif isinstance(module, nn.MultiheadAttention):
                nn.init.xavier_uniform_(module.in_proj_weight)
                nn.init.xavier_uniform_(module.out_proj.weight)
                if module.in_proj_bias is not None:
                    nn.init.constant_(module.in_proj_bias, 0.0)
                if module.out_proj.bias is not None:
                    nn.init.constant_(module.out_proj.bias, 0.0)

    def forward(self, x: torch.Tensor):
        """
        x: (batch, 2, num_classes)  # 两个集合，每个集合是 num_classes 维的数量分布
        返回: logits (batch, 2)
        """
        batch_size = x.size(0)

        # Log-compress counts for stability
        x = torch.log1p(x.clamp_min(0.0))

        # Encode counts: (batch, 2, num_classes, 1) → (batch, 2, num_classes, hidden_dim)
        set1 = self.count_encoder(x[:, 0].unsqueeze(-1))  # (batch, num_classes, hidden_dim)
        set2 = self.count_encoder(x[:, 1].unsqueeze(-1))

        # Add class embeddings (broadcast over batch)
        class_indices = torch.arange(self.num_classes, device=x.device)
        class_emb = self.class_emb(class_indices)  # (num_classes, hidden_dim)
        set1 = set1 + class_emb.unsqueeze(0)
        set2 = set2 + class_emb.unsqueeze(0)

        # Residual cross-attention with LayerNorm
        attn1, _ = self.cross_attn(query=set1, key=set2, value=set2)
        set1 = self.cross_norm1(set1 + attn1)
        attn2, _ = self.cross_attn(query=set2, key=set1, value=set1)
        set2 = self.cross_norm2(set2 + attn2)

        # PMA pooling with residual
        pool_query = self.pool_seed.expand(batch_size, -1, -1)
        pooled1, _ = self.pool(query=pool_query, key=set1, value=set1)
        set1_vec = self.pool_norm1(pool_query + pooled1).squeeze(1)
        pooled2, _ = self.pool(query=pool_query, key=set2, value=set2)
        set2_vec = self.pool_norm2(pool_query + pooled2).squeeze(1)

        # Antisymmetric comparator: 保证交换两队顺序 → 输出翻转
        combined = set1_vec - set2_vec  # (batch, hidden_dim)
        return self.comparator(combined)
