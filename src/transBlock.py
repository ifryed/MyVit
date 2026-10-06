import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbedding(nn.Module):
    def __init__(self, patch_shape=(4,4),channels=3, embedding_dim=128):
        super(PatchEmbedding, self).__init__()
        self.patch_shape = patch_shape
        self.channels = channels
        self.projection = nn.Linear(patch_shape[0]*patch_shape[1]*channels, embedding_dim)
        self.position_embedding = nn.Parameter(torch.rand(1,patch_shape[0]*patch_shape[1],embedding_dim))

    def forward(self, x, position=None):
        if x.dim() == 3:
            x = x.unsqueeze(0)
        B, C, H, W = x.shape
        ph, pw = self.patch_shape
        n_h, n_w = H // ph, W // pw
        x = x.reshape(B, C, n_h, ph, n_w, pw)
        x = x.permute(0, 2, 4, 1, 3, 5).reshape(B, n_h * n_w, C * ph * pw)
        x = self.projection(x)
        if position is not None:
            x = x + self.position_embedding[:, position]
        return x
    
class Attention(nn.Module):
    def __init__(self, embedding_dim):
        super(Attention, self).__init__()
        self.embedding_dim = embedding_dim
        self.Wq = nn.Linear(embedding_dim, embedding_dim)
        self.Wk = nn.Linear(embedding_dim, embedding_dim)
        self.Wv = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x):
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)
        return torch.softmax(Q @ K.T/torch.sqrt(self.embedding_dim)) @ V

class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.atten_heads = nn.ModuleList([Attention(self.head_dim) for _ in range(num_heads)])
        
        self.qkv = nn.Linear(embedding_dim, embedding_dim * 3)
        self.proj = nn.Linear(embedding_dim, embedding_dim)
        
    def forward(self, x):
        B, N, D = x.shape
        qkv = self.qkv(x)
        
        qkv = qkv.reshape(B,N,3,self.num_heads,self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        Q, K, V = qkv[0], qkv[1], qkv[2]
        
        scores = (Q @ K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention = scores.softmax(dim=-1)
        out = attention @ V
        
        out = out.permute(0, 2, 1, 3).contiguous()
        out = out.reshape(B, N, D)
        return self.proj(out), attention
        

class MLP(nn.Module):
    def __init__(self, embedding_dim, hidden_ratio=4):
        super(MLP, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_ratio = hidden_ratio
        self.fc1 = nn.Linear(embedding_dim, embedding_dim*hidden_ratio)
        self.fc2 = nn.Linear(embedding_dim*hidden_ratio, embedding_dim)
        
    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

class TrasnformerBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, hidden_ratio=4):
        super(TrasnformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.multi_head_attention = MultiHeadAttention(embedding_dim, num_heads)
        self.mlp = MLP(embedding_dim, hidden_ratio=hidden_ratio)
        
    def forward(self, in_x):
        
        x = self.norm1(in_x)
        x_att, attention = self.multi_head_attention(x)
        x = x + x_att
        
        out = x + self.mlp(x)
        return out, attention
    
    