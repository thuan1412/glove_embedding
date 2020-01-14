import torch
import numpy as np
from collections import Counter, defaultdict
import json

import time


class GloveDataset:
    def __init__(self, text, n_words=200000, window_size=5, batch_size=256):
        self._window_size = window_size
        self._tokens = text.split(" ")[:n_words]
        self._batch_size = batch_size
        word_counter = Counter()
        # words occurrence represented by Counter object
        word_counter.update(self._tokens)
        # word to id mapping in the descending order or occurrence
        self._word2id = {
            w: i
            for i, (w, _) in enumerate(word_counter.most_common())
        }
        self._id2word = {i:w for w, i in self._word2id.items()}
        self._vocab_len = len(self._word2id)

        self._id_tokens = [self._word2id[w] for w in self._tokens]

        self._create_coocurrence_matrix()

        self.save_mapping_dicts()

        print(f"# of words: {len(self._tokens)}")
        print(f"Vocabulary length:: {self._vocab_len}")

    def _create_coocurrence_matrix(self):
        cooc_mat = defaultdict(Counter)

        for i, w in enumerate(self._id_tokens):
            start_i = max(i - self._window_size, 0)
            end_i = min(i + self._window_size + 1, len(self._id_tokens))

            for j in range(start_i, end_i):
                if i != j:
                    c = self._id_tokens[j]
                    cooc_mat[w][c] += 1 / abs(j - i)

        self._i_idx = list()
        self._j_idx = list()
        self._xij = list()

        # Create indexes and tensor X
        for w, cnt in cooc_mat.items():
            for c, x in cnt.items():
                self._i_idx.append(w)
                self._j_idx.append(c)
                self._xij.append(x)

        # Convert indexes and value of tensor X to torch.Tensor
        if torch.cuda.is_available():
            self._i_idx = torch.LongTensor(self._i_idx).cuda()
            self._j_idx = torch.LongTensor(self._j_idx).cuda()
            self._xij = torch.FloatTensor(self._xij).cuda()
        else:
            self._i_idx = torch.LongTensor(self._i_idx)
            self._j_idx = torch.LongTensor(self._j_idx)
            self._xij = torch.FloatTensor(self._xij)

    def save_mapping_dicts(self):
        word_id_mappinng = {}
        word_id_mappinng['word2id'] = self._word2id
        word_id_mappinng['id2word'] = self._id2word
        with open("mapping.json", "w") as fp:
            json.dump(word_id_mappinng, fp)

    def get_batches(self, batch_size):
        #Generate random idx
        rand_ids = torch.LongTensor(np.random.choice(len(self._xij), len(self._xij), replace=False))

        for p in range(0, len(rand_ids), batch_size):
            batch_ids = rand_ids[p:p+batch_size]
            yield self._xij[batch_ids], self._i_idx[batch_ids], self._j_idx[batch_ids]

if __name__ == "__main__":
    start = time.time()
    dataset = GloveDataset(open("./text8").read(), 10000000)
    print("Elapsed tiem: ", time.time() - start)
