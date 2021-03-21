import torch 
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from model.base_model import BaseModel
from model.block import Block
from model.attention import MultiHeadAttentionLayer
from model.embedding import Embeddings
from model.feed_forward import FeedForward
from utils import constant
from utils.util import convert_adj

class GCNClassifier(BaseModel):
    def __init__(self, num_classes, n_mlp_layers, pooling,
                    vocab_size, emb_dim, pos_dim, n_blocks,
                    rnn, rnn_dim, n_rnn_layers, rnn_dropout, 
                    h_dim, n_heads, n_sub_layers_first, 
                    n_sub_layers_second, device, dropout=0.1, emb_dropout=0.5,
                    self_loop=True, emb_matrix=None, topn=5, **kwargs):
        super().__init__()
        self.device = device
        self.pooling = pooling
        self.aggcn = AGGCN(vocab_size, emb_dim, pos_dim, n_blocks,
                    rnn, rnn_dim, n_rnn_layers, rnn_dropout, 
                    h_dim, n_heads, n_sub_layers_first, 
                    n_sub_layers_second, device, dropout, emb_dropout,
                    self_loop, emb_matrix, topn)

        self.feed_forward = FeedForward(h_dim, n_mlp_layers)

        self.classifier = nn.Linear(h_dim, num_classes)
    
    def forward(self, inputs):
        words, masks, pos, deprel, head, subj_pos, obj_pos = inputs

        l = (masks.data.cpu().numpy() == 0).astype(np.int64).sum(1)
        maxlen = max(l)
        adj = [convert_adj(head_list.tolist(), maxlen, directed=False).reshape(1, maxlen, maxlen) for head_list in head]
        adj = np.concatenate(adj, axis=0)
        adj = Variable(torch.from_numpy(adj).to(self.device))
        h, pool_mask = self.aggcn(adj, inputs)
                # h.shape = (batch_size, mem_dim)

        # pooling
        subj_mask, obj_mask = subj_pos.eq(0).eq(0).unsqueeze(2), obj_pos.eq(0).eq(0).unsqueeze(2)  # invert mask
        h_out = pool(h, pool_mask, type= self.pooling)
        # h_out.shape = (batch_size, mem_dim)
        subj_out = pool(h, subj_mask, type="max")
        # subj_out.shape = (batch_size, mem_dim)
        obj_out = pool(h, obj_mask, type="max")
        outputs = torch.cat([h_out, subj_out, obj_out], dim=1)
        # outpus.shape = (batch_size, mem_dim*3)
        outputs = self.feed_forward(outputs)
        outputs = self.classifier(outputs)
        return outputs, h_out



class AGGCN(BaseModel):
    def __init__(self, vocab_size, emb_dim, pos_dim, n_blocks,
                    rnn, rnn_dim, n_rnn_layers, rnn_dropout, 
                    h_dim, n_heads, n_sub_layers_first,
                    n_sub_layers_second, device, dropout=0.1, emb_dropout=0.5,
                    self_loop=True, emb_matrix=None, topn=5):
        super().__init__()

        self.embedding = Embeddings(vocab_size, emb_dim, pos_dim, device, emb_dropout, emb_matrix, topn)
        
        self.in_dim = emb_dim + pos_dim
        self.rnn_dim = rnn_dim
        self.n_rnn_layers = n_rnn_layers
        self.rnn_dropout = nn.Dropout(rnn_dropout)
        self.device = device
        self.rnn = rnn
        self.n_blocks = n_blocks
        if rnn:
            self.fc_rnn = nn.Linear(self.in_dim, rnn_dim)
            self.rnn_eoncder = nn.LSTM(rnn_dim, rnn_dim, n_rnn_layers, 
                                        batch_first=True, dropout=rnn_dropout, bidirectional=True)

            self.in_dim = self.rnn_dim * 2
        
        self.fc = nn.Linear(self.in_dim, h_dim)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList()
        self.attention_guide_layer = MultiHeadAttentionLayer(h_dim, n_heads, dropout, device)
        for i in range(self.n_blocks):
            if i == 0:
                self.blocks.append(Block(h_dim, n_heads, n_sub_layers_first, n_sub_layers_second,
                                    device, None, dropout, self_loop))
            else:
                self.blocks.append(Block(h_dim, n_heads, n_sub_layers_first, n_sub_layers_second,
                                    device, self.attention_guide_layer, dropout, self_loop))

        self.aggregate_W = nn.Linear(2 * self.n_blocks * h_dim, h_dim)

    def forward(self, adj, inputs):
        words, masks, pos, deprel, head, subj_pos, obj_pos = inputs 
        src_mask = (words != constant.PAD_ID).unsqueeze(-2)

        embs = self.embedding(words, pos)

        if self.rnn:
            embs = self.fc_rnn(embs)
            gcn_inputs = self.rnn_dropout(self.encode_with_rnn(embs, masks, words.shape[0]))

        else:
            gcn_inputs = embs
        gcn_inputs = self.fc(gcn_inputs)

        layer_list = []
        outputs = gcn_inputs
        for i in range(self.n_blocks):
            layer_list.extend(self.blocks[i](adj, masks, outputs))
            outputs = layer_list[-1]

        aggregate_out = torch.cat(layer_list, dim=2).to(self.device)
        dcgcn_output = self.aggregate_W(aggregate_out)
        mask = (adj.sum(2) + adj.sum(1)).eq(0).unsqueeze(2)
        return dcgcn_output, mask

    def encode_with_rnn(self, rnn_inputs, masks, batch_size):
        seq_lens = list(masks.data.eq(constant.PAD_ID).long().sum(1).squeeze())
        h0, c0 = rnn_zero_state(batch_size,self.rnn_dim, self.n_rnn_layers, self.device)
        rnn_inputs = nn.utils.rnn.pack_padded_sequence(rnn_inputs, seq_lens, batch_first=True)
        rnn_outputs, (ht, ct) = self.rnn_eoncder(rnn_inputs, (h0, c0))
        rnn_outputs, _ = nn.utils.rnn.pad_packed_sequence(rnn_outputs, batch_first=True)
        return rnn_outputs


def rnn_zero_state(batch_size, hidden_dim, num_layers, device, bidirectional=True):
    total_layers = num_layers * 2 if bidirectional else num_layers
    state_shape = (total_layers, batch_size, hidden_dim)
    h0 = c0 = Variable(torch.zeros(*state_shape), requires_grad=False).to(device)
    return h0, c0


def pool(h, mask, type='max'):
    if type == 'max': 
        h = h.masked_fill(mask, -1e11) #对mask部分取无穷小，保证不会取到
        return torch.max(h, 1)[0]
    elif type == 'avg':
        h = h.masked_fill(mask, 0) #保证mask不影响计算
        return h.sum(1) / (mask.size(1) - mask.float().sum(1)) # 
    else:# type=='sum'
        h = h.masked_fill(mask, 0)
        return h.sum(1)