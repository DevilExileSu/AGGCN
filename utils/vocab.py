import numpy as np
import pickle
import os
import random

from utils import constant



class Vocab(object):
    def __init__(self, filename, load=False, word_counter=None, threshold=0):
        if load:
            assert os.path.exists(filename), "Vocab file does not exist at " + filename
            # load from file and ignore all other params
            self.id2word, self.word2id = self.load(filename)
            self.size = len(self.id2word)
            print("Vocab size {} loaded from file".format(self.size))
        else:
            print("Creating vocab from scratch...")
            assert word_counter is not None, "word_counter is not provided for vocab creation."
            self.word_counter = word_counter
            if threshold > 1:
                # remove words that occur less than thres
                self.word_counter = dict([(k,v) for k,v in self.word_counter.items() if v >= threshold])
            self.id2word = sorted(self.word_counter, key=lambda k:self.word_counter[k], reverse=True)
            # add special tokens to the beginning
            self.id2word = [constant.PAD_TOKEN, constant.UNK_TOKEN] + self.id2word
            self.word2id = dict([(self.id2word[idx],idx) for idx in range(len(self.id2word))])
            self.size = len(self.id2word)
            self.save(filename)
            print("Vocab size {} saved to file {}".format(self.size, filename))

    def load(self, filename):
        with open(filename, 'rb') as infile:
            id2word = pickle.load(infile)
            word2id = dict([(id2word[idx], idx) for idx in range(len(id2word))])
        return id2word, word2id

    def save(self, filename):
        if os.path.exists(filename):
            print("Overwriting old vocab file at " + filename)
            os.remove(filename)
        with open(filename, 'wb') as outfile:
            pickle.dump(self.id2word, outfile)
        return
