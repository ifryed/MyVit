import torch
import torch.nn as nn
from transBlock import PatchEmbedding, TrasnformerBlock

class ViT(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, embedding_dim, num_heads, num_layers, dropout=0.1):
        super(ViT, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.hidden_ratio = 4
        
        # Learnable CLS token
        self.cls_token = nn.Parameter(
            torch.randn(1, 1, embedding_dim)
        )
        
        self.patch_embedding = PatchEmbedding(patch_shape=(patch_size, patch_size), embedding_dim=embedding_dim) 
        # +1 for CLS token
        self.pos_embed = nn.Parameter(
            torch.randn(
                1,
                self.num_patches + 1,
                embedding_dim
            )
        )
        self.transformer_blocks = nn.ModuleList(
            [TrasnformerBlock(embedding_dim, num_heads, self.hidden_ratio) for _ in range(num_layers)]
            )
        self.classifier = nn.Linear(embedding_dim, num_classes)
        self.drop = nn.Dropout(dropout)
        
    def forward(self, x):
        B, C, H, W = x.shape
        assert H == W, "Image must be square"
        
        # Embed the patches
        x = self.patch_embedding(x)
        
        # CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        
        x = torch.cat(
            [cls_tokens, x],
            dim=1
        )
        
        # Add positional embedding
        x = x + self.pos_embed
        x = self.drop(x)
        
        attentions = []
        for t_block in self.transformer_blocks:
            x, attention = t_block(x)
            attentions.append(attention)
        cls_token_out = x[:, 0]
        
        return self.classifier(cls_token_out), attentions