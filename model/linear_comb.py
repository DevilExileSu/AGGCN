import torch.nn as nn
from model.base_model import BaseModel

class LinearCombinationLayer(BaseModel):
    """
    input:
        embeddings
    return:
        embedding
    """
    def __init__(self, h_dim, n_heads):
        super().__init__()
        self.h_dim = h_dim
        self.n_heads = n_heads
        
        self.linear = nn.Linear(self.h_dim * n_heads, self.h_dim)
    
    def forward(self, inputs):

        output = self.linear(inputs)
        return output