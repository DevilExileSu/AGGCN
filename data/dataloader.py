import json
import torch
import numpy as np
from utils import constant
class BaseDataLoader(object):
    """
    Nonuse torch.utils.data
    """
    def __init__(self, filename, batch_size, shuffle, logger):
        """
        Initialization data file path, batch data size, shuffle data
        Read data from data file
        Preprocess the data
        Spilt the data according to batch_size
        """
        pass
    def __len__(self):
        """
        How many batch
        """
        raise NotImplementedError
    def __getitem__(self, index):
        """
        Return batch_size data pairs
        """
        raise NotImplementedError
    def __read_data(self,):
        pass
    def __preprocess_data(self,):
        pass

    
class SemevalDataLoader(BaseDataLoader):
    def __init__(self, filename, batch_size, vocab,  logger, word_dropout=0.01, eval=False, lower=True):
        self.batch_size = batch_size
        self.filename = filename
        self.vocab = vocab
        self.eval= eval
        self.word_dropout = word_dropout
        self.logger = logger
        self.lower = lower
        self.label2id = constant.LABEL_TO_ID
        self.id2label = dict([(v,k) for k,v in self.label2id.items()])
        data = self.__read_data()

        data = [data[i:i+batch_size] for i in range(0, len(data), batch_size)]
        self.data = data
        self.logger.debug("{} batches created for {}".format(len(data), filename))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if not isinstance(index, int):
                raise TypeError
        if index < 0 or index >= len(self.data):
            raise IndexError
        batch = self.data[index]
        batch_size = len(batch)
        batch = list(zip(*batch))
        # sort all fields by lens for easy RNN operations
        sorted_idx = sorted(range(len(batch[0])), key=lambda i:len(batch[0][i]), reverse=True)
        orig_idx = [sorted_idx.index(i) for i,x in enumerate(sorted_idx)]

        if not self.eval:
            words = [word_dropout(sent, self.word_dropout) for sent in batch[0]]
        else:
            words = [sent for sent in batch[0]]
        words = get_long_tensor(words, batch_size, sorted_idx)
        masks = torch.eq(words, 0)
        pos = get_long_tensor(batch[1], batch_size, sorted_idx)
        deprel = get_long_tensor(batch[2], batch_size, sorted_idx)
        head = get_long_tensor(batch[3], batch_size, sorted_idx)
        subj_positions = get_long_tensor(batch[4], batch_size, sorted_idx)
        obj_positions = get_long_tensor(batch[5], batch_size, sorted_idx)
        rels = torch.LongTensor(batch[6])[sorted_idx]
        return (words, masks, pos, deprel, head, subj_positions, obj_positions, orig_idx, rels)   


    def __read_data(self):
        self.logger.debug("-----------read data-----------")
        with open(self.filename) as f:
            data = json.load(f)
        if not self.eval:
            idx = np.random.permutation(len(data))
            data = [data[i] for i in idx]

        self.logger.debug("{} has data {}".format(self.filename, len(data)))

        return self.__preprocess_data(data)
    
    def __preprocess_data(self, data):
        processed = []
        for d in data:
            tokens = list(d['token'])
            if self.lower:
                tokens = [t.lower() for t in tokens]
            # anonymize tokens
            ss, se = d['subj_start'], d['subj_end']
            os, oe = d['obj_start'], d['obj_end']

            tokens = map_to_ids(tokens, self.vocab.word2id)

            pos = map_to_ids(d['stanford_pos'], constant.POS_TO_ID)
            deprel = map_to_ids(d['stanford_deprel'], constant.DEPREL_TO_ID)
            head = [int(x) for x in d['stanford_head']]
            assert any([x == 0 for x in head])
            l = len(tokens)
            subj_positions = get_positions(d['subj_start'], d['subj_end'], l)
            obj_positions = get_positions(d['obj_start'], d['obj_end'], l)
            relation = self.label2id[d['relation']]
            processed += [(tokens, pos, deprel, head, subj_positions, obj_positions, relation)]

        return processed



def map_to_ids(tokens, vocab):
    ids = [vocab[t] if t in vocab else constant.UNK_ID for t in tokens]
    return ids

def get_positions(start_idx, end_idx, length): #转化成序列
    """ Get subj/obj position sequence. """
    return list(range(-start_idx, 0)) + [0]*(end_idx - start_idx + 1) + list(range(1, length-end_idx))


def get_long_tensor(tokens_list, batch_size, sorted_idx=None): # 用PAD_ID填充token为该batch中最大的 
    
    """ Convert list of list of tokens to a padded LongTensor. """
    token_len = max(len(x) for x in tokens_list)
    tokens = torch.LongTensor(batch_size, token_len).fill_(constant.PAD_ID)

    for i, s in enumerate(tokens_list):
        tokens[i, :len(s)] = torch.LongTensor(s)
    if sorted_idx is not None:
        return tokens[sorted_idx]
    else:
        return tokens

def word_dropout(tokens, dropout):
    """ Randomly dropout tokens (IDs) and replace them with <UNK> tokens. """
    return [constant.UNK_ID if x != constant.UNK_ID and np.random.random() < dropout \
            else x for x in tokens]