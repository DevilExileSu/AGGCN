import torch 
import torch.nn as nn
from model.base_model import BaseModel

class FeedForward(BaseModel):
    def __init__(self, h_dim, n_mlp_layers):
        super().__init__()
        in_dim = h_dim * 3
        layers = [nn.Linear(in_dim, h_dim), nn.ReLU()]
        for i in range(n_mlp_layers-1):
            layers += [nn.Linear(h_dim, h_dim), nn.ReLU()]
        self.out_mlp = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.out_mlp(inputs)