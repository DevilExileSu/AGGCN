import torch
import torch.nn as nn
import torch.nn.functional as F
from model.base_model import BaseModel
from model.linear_comb import LinearCombinationLayer

class SingleDCGCN(BaseModel):
    def __init__(self, h_dim, n_sub_layers, device, dropout=0.1, self_loop=True):
        super().__init__()
        self.h_dim = h_dim
        self.n_sub_layers = n_sub_layers
        assert h_dim % n_sub_layers == 0
        self.sub_dim = self.h_dim // self.n_sub_layers
        
        self.dropout = nn.Dropout(dropout)
        self.self_loop = self_loop
        self.sub_layers = nn.ModuleList()
        self.device = device
        
        for j in range(n_sub_layers):
            self.sub_layers.append(nn.Linear(self.h_dim + self.sub_dim * j, self.sub_dim))
    
        self.linear_combination_layer = LinearCombinationLayer(h_dim, n_heads=1)

    def forward(self, adj, inputs):
        denom = adj.sum(2).unsqueeze(2) + 1
        outputs = inputs
        cache_list = [outputs]
        output_list = []
        for i in range(self.n_sub_layers):
            Ax = adj.bmm(outputs)
            AxW = self.sub_layers[i](Ax)
            if self.self_loop:
                AxW = AxW + self.sub_layers[i](outputs)
            AxW = AxW / denom
            gAxW = F.relu(AxW)
            cache_list.append(gAxW)
            outputs = torch.cat(cache_list, dim=2)
            #outputs.shape = (batch_size, maxlen, mem_dim + sub_layers * i)
            # AxW.shape = (batch_size, maxlen, sub_dim)
            output_list.append(self.dropout(gAxW)) 

        output = torch.cat(output_list, dim=2).to(self.device)
        # output = (batch_size, maxlen, h_dim)
        output += inputs
        output = self.linear_combination_layer(output)
        return output

class MultiDCGCN(BaseModel):
    def __init__(self, h_dim, n_sub_layers, n_heads, device, dropout=0.1, self_loop=True):
        super().__init__()
        self.h_dim = h_dim
        self.n_sub_layers = n_sub_layers
        self.n_heads = n_heads
        assert h_dim % n_sub_layers == 0
        self.sub_dim = self.h_dim // self.n_sub_layers
        
        self.dropout = nn.Dropout(dropout)
        self.self_loop = self_loop
        self.device = device
        
        self.sub_layers = nn.ModuleList()
        for i in range(n_heads):
            for j in range(n_sub_layers):
                self.sub_layers.append(nn.Linear(self.h_dim + self.sub_dim * j, self.sub_dim))
        
        self.linear_combination_layer = LinearCombinationLayer(h_dim, n_heads = n_heads)

    def forward(self, adj_list, inputs):
        multi_head_list = []
        for i in range(self.n_heads):
            adj = adj_list[i]
            denom = adj.sum(2).unsqueeze(2) + 1
            outputs = inputs
            cache_list = [outputs]
            output_list = []

            for j in range(self.n_sub_layers):
                idx = i * self.n_sub_layers + j
                Ax = adj.bmm(outputs)
                AxW = self.sub_layers[idx](Ax)
                if self.self_loop:
                    AxW = AxW + self.sub_layers[idx](outputs)
                AxW = AxW / denom
                gAxW = F.relu(AxW)
                cache_list.append(gAxW)
                outputs = torch.cat(cache_list, dim=2)
                #outputs.shape = (batch_size, maxlen, mem_dim + head_dim * i)
                # AxW.shape = (batch_size, maxlen, head_dim)
                output_list.append(self.dropout(gAxW))
            output = torch.cat(output_list, dim=2).to(self.device)
            output += inputs

            multi_head_list.append(output)
        multi_output = torch.cat(multi_head_list, dim=2).to(self.device)
        # [batch_size, maxlen, mem_dim * n_heads]
        output = self.linear_combination_layer(multi_output)

        return output