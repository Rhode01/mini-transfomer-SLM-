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
