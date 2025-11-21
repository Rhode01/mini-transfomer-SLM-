import torch
import torch.nn as nn
import math

class SelfAttention(nn.Module):
    def __init__(self, d_model) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_k = d_model  

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
    def forward(self, x):
        

        Q = self.W_q(x) 
        K = self.W_k(x)  
        V = self.W_v(x) 
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        weights = torch.softmax(scores, dim=-1)

        output = torch.matmul(weights, V) 

        return output, weights
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()

        assert d_model % num_heads == 0, "d_model must divide num_heads"

        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Combined linear layers for Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # Output projection
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch, seq_len, d_model = x.shape

        # 1. Linear projections
        Q = self.W_q(x)  # (B, S, d_model)
        K = self.W_k(x)
        V = self.W_v(x)

        # 2. Split into heads
        Q = Q.view(batch, seq_len, self.num_heads, self.d_k)
        K = K.view(batch, seq_len, self.num_heads, self.d_k)
        V = V.view(batch, seq_len, self.num_heads, self.d_k)

        # Rearrange to put heads on batch dimension
        Q = Q.transpose(1, 2)  # (B, heads, S, d_k)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # 3. Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(weights, V)  # (B, heads, S, d_k)

        # 4. Combine heads back
        output = output.transpose(1, 2).contiguous()  # (B, S, heads, d_k)
        output = output.view(batch, seq_len, d_model)

        # 5. Output projection
        return self.W_o(output), weights

