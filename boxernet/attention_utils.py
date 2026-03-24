# pyre-unsafe
import torch
import torch.nn as nn

"""
Modified from nn_utils.py to reduce dim_head and cleaned up.
Includes AlternatingAttentionBlock for VGGT-style frame/global attention.
"""


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head**-0.5
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, y, attn_mask=None):
        # Pre-norm (apply norm before attention) as in https://arxiv.org/abs/2002.04745.
        x = self.norm(x)
        y = self.norm(y)
        # Compute Q,K,V matrices.
        q = self.to_q(x)
        k = self.to_k(y)
        v = self.to_v(y)
        # Split into separate heads.
        B, N_q, _ = q.shape
        N_kv = k.shape[1]
        q = q.view(B, N_q, self.heads, -1).transpose(1, 2)
        k = k.view(B, N_kv, self.heads, -1).transpose(1, 2)
        v = v.view(B, N_kv, self.heads, -1).transpose(1, 2)
        # Compute softmax(Q*K^T / sqrt(D))*V
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask)
        # Combine the multi-headed output.
        out = out.transpose(1, 2).contiguous().view(B, N_q, -1)
        return self.to_out(out)


class AttentionBlockV2(nn.Module):
    def __init__(self, dim=256, depth=6, heads=4, mlp_mult=4):
        """Vanilla transformer implementation.
        Inputs:
          dim - dimensionality of attention and mlp (shared for simplicity)
          depth - number of layers
          heads - number of attention heads
        """
        super().__init__()
        assert dim % heads == 0
        dim_head = int(dim // heads)
        dim_mlp = dim * mlp_mult
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(
                            dim,
                            heads=heads,
                            dim_head=dim_head,
                        ),
                        FeedForward(dim, dim_mlp),
                    ]
                )
            )

    def forward(self, x, y=None, attn_mask=None):
        """If y not provided this is self-attention; otherwise is cross-attention.

        For cross-attention, the queries should be independent, so changing the value of
        one query should not affect the other queries.

        Inputs:
            x: BxNxD batch of query tokens, D is dimenionality of each token (maps to Q)
            y: optional BxMxD batch input tokens, D is dim of each token (maps to K, V)
        Outputs:
            out: BxNxD tensor of transformed queries, same shape as x

        """
        if y is None:
            y = x  # Self-attention.
        # else: # Cross-attention.

        for attn, ff in self.layers:
            x = attn(x, y, attn_mask=attn_mask) + x
            x = ff(x) + x
        return x
