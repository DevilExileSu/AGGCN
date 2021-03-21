import torch
from model.base_model import BaseModel
from model.densely_graph_conv import SingleDCGCN, MultiDCGCN

class Block(BaseModel):
    def __init__(self, h_dim, n_heads, n_sub_layers_first, n_sub_layers_second, device, attention_guide_layer, dropout=0.1, self_loop=True):
        super().__init__()
        self.h_dim = h_dim
        # self.n_heads = n_heads
        self.n_sub_layers_first = n_sub_layers_first
        self.n_sub_layers_second = n_sub_layers_second
        self.device = device
        self.attention_guide_layer = attention_guide_layer

        if self.attention_guide_layer is not None:
            # self.attention_guide_layer = MultiHeadAttentionLayer(h_dim, n_heads, dropout, device)
            self.densely_connect_layer_first = MultiDCGCN(h_dim, n_sub_layers_first, n_heads, device, dropout, self_loop)
            self.densely_connect_layer_second = MultiDCGCN(h_dim, n_sub_layers_second, n_heads, device, dropout, self_loop)
        else:
            self.attention_guide_layer = None
            self.densely_connect_layer_first = SingleDCGCN(h_dim, n_sub_layers_first, device, dropout, self_loop)
            self.densely_connect_layer_second = SingleDCGCN(h_dim, n_sub_layers_second, device, dropout, self_loop)
        
    
    def forward(self, adj, mask, inputs):

        if self.attention_guide_layer is not None:
            _, attention_guide_adjs = self.attention_guide_layer(inputs, inputs, inputs, mask)
            
            attention_guide_adjs = [attn_adj.squeeze(1) for attn_adj in torch.split(attention_guide_adjs, 1, dim=1)]
            
            output_first = self.densely_connect_layer_first(attention_guide_adjs, inputs)
            output_second = self.densely_connect_layer_second(attention_guide_adjs, output_first)

        else:
            output_first = self.densely_connect_layer_first(adj, inputs)
            output_second = self.densely_connect_layer_second(adj, output_first)

        return [output_first, output_second]

