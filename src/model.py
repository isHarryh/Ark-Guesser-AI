import torch
import torch.nn as nn
import torch.nn.functional as F


class ArkGuesserModelV0(nn.Module):
    """
    输入: x, shape (batch, 2, num_classes)
    输出: logits, shape (batch, 2)  # 未 softmax 的 logits
    架构:
      1. 编码器: 对每个集合的数量分布做编码 (共享MLP + 类别嵌入)
      2. 双向交叉注意力: 集合间交互
      3. 池化: 聚合为集合向量 (可学习加权平均池化)
      4. 比较器: 输出胜者 logits
    """

    def __init__(
        self,
        num_classes: int,
        hidden_dim: int = 256,
        attn_heads: int = 8,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.n_heads = attn_heads

        # Class embedding
        self.class_emb = nn.Embedding(num_classes, hidden_dim)

        # Shared MLP encoder
        self.shared_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Cross-attention layers
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=attn_heads,
            dropout=dropout,
            add_zero_attn=True,
            batch_first=True,
        )

        # Comparator
        self.comparator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

        # Xavier initialization
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.Embedding):
                nn.init.xavier_uniform_(module.weight)
            elif isinstance(module, nn.MultiheadAttention):
                nn.init.xavier_uniform_(module.in_proj_weight)
                nn.init.xavier_uniform_(module.out_proj.weight)
                nn.init.constant_(module.in_proj_bias, 0.0)
                nn.init.constant_(module.out_proj.bias, 0.0)

    def forward(self, x: torch.Tensor):
        """
        x: (batch, 2, num_classes)  # 两个集合，每个集合是num_classes维的数量分布
        返回: logits (batch, 2)  # 未 softmax 的 logits
        """
        batch_size = x.size(0)
        # 类别嵌入 (batch, 2, num_classes, hidden_dim)
        class_indices = (
            torch.arange(self.num_classes, device=x.device)
            .unsqueeze(0)
            .unsqueeze(0)
            .expand(batch_size, 2, self.num_classes)
        )
        class_emb = self.class_emb(class_indices)
        # 编码两个组的数量分布
        set1 = self.shared_encoder(x[:, 0].unsqueeze(-1))  # (batch, num_classes, hidden_dim)
        set2 = self.shared_encoder(x[:, 1].unsqueeze(-1))
        # 融合类别嵌入和数量编码
        set1 = set1 + class_emb[:, 0]
        set2 = set2 + class_emb[:, 1]
        # 交叉注意力
        set1_attn, _ = self.cross_attn(query=set1, key=set2, value=set2)
        set2_attn, _ = self.cross_attn(query=set2, key=set1, value=set1)
        set1, set2 = set1_attn, set2_attn
        # 池化
        set1_pooled = set1.mean(dim=1)  # (batch, hidden_dim)
        set2_pooled = set2.mean(dim=1)  # (batch, hidden_dim)
        # 送入比较器
        # combined = torch.cat([set1_pooled, set2_pooled], dim=-1)  # (batch, hidden_dim * 2)
        combined = set1_pooled - set2_pooled  # (batch, hidden_dim)
        log_probs = self.comparator(combined)  # (batch, 2)
        return log_probs
