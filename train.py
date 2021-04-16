import torch
import argparse
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from config import Config,Logger
from utils.util import get_optimizer, initialize_weights
from utils.vocab import Vocab
from data import SemevalDataLoader
from model.model import GCNClassifier
from utils import constant
from trainer.gcn_trainer import GCNTrainer
parser = argparse.ArgumentParser()

# dataset parameter
parser.add_argument('--train_data', type=str, default='datasets/train.json')
parser.add_argument('--valid_data', type=str, default='datasets/test.json')
parser.add_argument('--vocab', type=str, default='datasets/vocab.pkl')
parser.add_argument('--embed_matrix', type=str, default='datasets/embedding.npy')
parser.add_argument('--lower', type=bool, default=True)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--word_dropout', type=float, default=0.04, help='The rate at which randomly set a word to UNK.')


# model parameter
parser.add_argument('--h_dim', type=int, default=360)
parser.add_argument('--emb_dim', type=int, default=300)
parser.add_argument('--pos_dim', type=int, default=30)
parser.add_argument('--n_mlp_layers', type=int, default=1)
parser.add_argument('--n_blocks', type=int, default=2)
parser.add_argument('--n_heads', type=int, default=3)
parser.add_argument('--topn', type=int, default=1e10)
parser.add_argument('--rnn', type=bool, default=True)
parser.add_argument('--rnn_dim', type=int, default=300)
parser.add_argument('--n_rnn_layers', type=int, default=1)
parser.add_argument('--rnn_dropout', type=float, default=0.5, help='The rate at which randomly set a word to UNK.')
parser.add_argument('--n_sub_layers_first', type=int, default=2)
parser.add_argument('--n_sub_layers_second', type=int, default=4, help='Num of the second sublayers in dcgcn block.')
parser.add_argument('--dropout', type=float, default=0.3, help='GCN layer dropout rate.')
parser.add_argument('--emb_dropout', type=float, default=0.3)
parser.add_argument('--self_loop', type=bool, default=True)
parser.add_argument('--pooling', choices=['max', 'avg', 'sum', 'self-att', 'cnn'], default='max', help='Pooling function type. Default max.')

# Loss function and Optimizer parameter
parser.add_argument('--lr', type=float, default=0.5)
parser.add_argument('--optimizer', choices=['sgd', 'adam', 'adamax'], default='sgd', help='Optimizer: sgd, adagrad, adam or adamax.')
parser.add_argument('--conv_l2', type=float, default=0, help='L2-weight decay on conv layers only.')
parser.add_argument('--pooling_l2', type=float, default=0.002, help='L2-penalty for all pooling output.')
parser.add_argument('--lr_decay', type=float, default=0.9, help='Learning rate decay rate.')
parser.add_argument('--lr_decay_patience', type=int, default=6)

# train parameter
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--save_dir', type=str, default='./saved_models')
parser.add_argument('--save_epochs', type=int, default=5, help='Save model checkpoints every k epochs.')
parser.add_argument('--early_stop', type=bool, default=True)
parser.add_argument('--patience', type=int, default=20)
parser.add_argument('--resume', type=bool, default=False)
parser.add_argument('--resume_path', type=str, default='./saved_models/model_best.pt')
parser.add_argument('--log_step', type=int, default=20)
parser.add_argument('--max_grad_norm', type=float, default=5.0, help='Gradient clipping.')

# other
parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
parser.add_argument('--config_file', type=str, default='./config.json')
parser.add_argument('--seed', type=int, default=1234)

args = parser.parse_args()
logger = Logger()

cfg = Config(logger=logger, args=args)
cfg.print_config()
cfg.save_config(cfg.config['config_file'])

torch.manual_seed(cfg.config['seed'])
torch.cuda.manual_seed(cfg.config['seed'])
torch.backends.cudnn.enabled = False
np.random.seed(cfg.config['seed'])


# vocab
vocab = Vocab(cfg.config['vocab'], load=True)
vocab_size = vocab.size
emb_matrix = np.load(cfg.config['embed_matrix'])
assert emb_matrix.shape[0] == vocab.size
assert emb_matrix.shape[1] == cfg.config['emb_dim']


# dataloader
# filename, batch_Size, vocab,  logger, word_dropout=0.01, eval=False, lower=True
data_loader = SemevalDataLoader(cfg.config['train_data'], cfg.config['batch_size'], vocab,
                                logger, cfg.config['word_dropout'], False, cfg.config['lower'])

valid_data_loader = SemevalDataLoader(cfg.config['valid_data'], cfg.config['batch_size'], vocab,
                                logger, cfg.config['word_dropout'], True, cfg.config['lower'])


# model 
"""
num_classes, n_mlp_layers, 
                    vocab_size, emb_dim, pos_dim, n_blocks,
                    rnn, rnn_dim, n_rnn_layers, rnn_dropout, 
                    h_dim, n_heads, n_sub_layers_first, 
                    n_sub_layers_second, device, dropout=0.1, 
                    self_loop=True, emb_matrix=None, topn=5,
"""
device = 'cuda:0' if cfg.config['cuda'] else 'cpu'
num_classes = len(constant.LABEL_TO_ID)
model = GCNClassifier(num_classes=num_classes,vocab_size=vocab_size, emb_matrix=emb_matrix, device=device, **cfg.config)
model.to(device)
logger.info(model)

# optimizer and criterion
param = [p for p in model.parameters() if p.requires_grad]
optimizer = get_optimizer(cfg.config['optimizer'], param, lr=cfg.config['lr'])
# optimizer = torch.optim.SGD(param, lr=cfg.config['lr'])
lr_scheduler = ReduceLROnPlateau(optimizer, 'max', factor=cfg.config['lr_decay'], patience=cfg.config['lr_decay_patience'])
criterion = nn.CrossEntropyLoss()


#trainer 
model.apply(initialize_weights)
trainer = GCNTrainer(model=model, optimizer=optimizer, criterion=criterion, cfg=cfg.config, logger=logger, 
                    data_loader=data_loader, valid_data_loader=valid_data_loader, lr_scheduler=lr_scheduler)


trainer.train()
