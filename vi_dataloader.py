import torch
import numpy as np
from collections import Counter, defaultdict
import json

import time

class ViDataLoader(object):
    def __init__(self, filename="cleaned_text.txt", n_lines=1000, window_size=5, min_occurrence=50):
        self._window_size = window_size
        self._n_lines = n_lines
        self._lines = open(filename, "r").readlines()[:n_lines]
        self._lines = [line.replace('\n', '') for line in self._lines]
        self._create_vocabulary()    
        self._create_coocurrence_matrix()

    def  _create_vocabulary(self):
        word_counter = Counter()
        for line in self._lines[:self._n_lines]:
            # line = line.replace("\n", '')
            word_counter.update(line.split(" "))
        self._word2id = {
            w: i
            for i, (w, _) in enumerate(word_counter.most_common())
        }
        self._id2word = {i:w for w, i in self._word2id.items()}
        self._vocab_len = len(self._word2id)
        self._save_dataloader_data()
        print("Vocab size: ", self._vocab_len)

    def _create_coocurrence_matrix(self):
        # first, make a defaultdict(Counter) for storing the matrix
        cooc_mat = defaultdict(Counter)
        for line in self._lines:
            line_id = [self._word2id[word] for word in line.split()]
            for i, w in enumerate(line_id):
                start_i = max(i-self._window_size, 0)
                end_id = min(i+self._window_size, len(line_id))
                for j in range(start_i, end_id):
                    if i != j:
                        c = line_id[j]
                        cooc_mat[w][c] += 1 / abs(j-i)

        # convert matrix into 3 vector( similar to sparse matrix)        
        
        self._i_idx = list()
        self._j_idx = list()
        self._xij = list()

        for w, cnt in cooc_mat.items():
            for c, x in cnt.items():
                self._i_idx.append(w)
                self._j_idx.append(c)
                self._xij.append(x)

        # convert above vectors to torch.Tensor
        if torch.cuda.is_available():
            self._i_idx = torch.LongTensor(self._i_idx).cuda()
            self._j_idx = torch.LongTensor(self._j_idx).cuda()
            self._xij = torch.FloatTensor(self._xij).cuda()
        else:
            self._i_idx = torch.LongTensor(self._i_idx)
            self._j_idx = torch.LongTensor(self._j_idx)
            self._xij = torch.FloatTensor(self._xij)
    
    def get_batches(self, batch_size):
        #Generate random idx
        rand_ids = torch.LongTensor(np.random.choice(len(self._xij), len(self._xij), replace=False))

        for p in range(0, len(rand_ids), batch_size):
            batch_ids = rand_ids[p:p+batch_size]
            yield self._xij[batch_ids], self._i_idx[batch_ids], self._j_idx[batch_ids]

    def _save_dataloader_data(self):
        data = {}
        data['word2id'] = self._word2id
        data['id2word'] = self._id2word
        data['vocab_size'] = self._vocab_len
        with open("mapping.json", "w") as fp:
            json.dump(data, fp)

dataset = ViDataLoader(filename="cleaned_text.txt", n_lines=1000, window_size=4, min_occurrence=50)