import torch
import torch.nn as nn
from model.base_model import BaseModel
from utils import constant

class Embeddings(BaseModel):
    def __init__(self, vocab_size, emb_dim, pos_dim, device, dropout, emb_matrix=None, topn=5):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_dim)
        self.pos_emb = nn.Embedding(len(constant.POS_TO_ID), pos_dim) if pos_dim > 0 else None
        self.device = device
        self.dropout = nn.Dropout(dropout)
        if emb_matrix is None:
            self.emb.weight.data[1:, :].uniform_(-1.0, 1.0)
        else:
            self.emb_matrix = torch.from_numpy(emb_matrix)
            self.emb.weight.data.copy_(self.emb_matrix)
        
        if topn <= 0:
            self.emb.weight.requires_grad = False
        elif topn < vocab_size:
            self.emb.weight.register_hook(lambda x: keep_partial_grad)

    def forward(self, words, pos):
        word_embs = self.emb(words)
        embs = [word_embs]

        if self.pos_emb is not None:
            embs += [self.pos_emb(pos)]
        
        embs = torch.cat(embs, dim=2).to(self.device)
        return self.dropout(embs)



def keep_partial_grad(grad, topk):

    assert topk < grad.size(0)
    grad.data[topk:].zero_()
    return grad

