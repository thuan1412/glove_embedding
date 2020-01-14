import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_


class GloveModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(GloveModel, self).__init__()
        self.wi = nn.Embedding(vocab_size, embedding_dim)
        self.wi.weight = xavier_normal_(self.wi.weight)

        self.wj = nn.Embedding(vocab_size, embedding_dim)
        self.wj.weight = xavier_normal_(self.wj.weight)

        self.bi = nn.Embedding(vocab_size, 1)
        self.bi.weight = xavier_normal_(self.bi.weight)

        self.bj = nn.Embedding(vocab_size, 1)
        self.bj.weight = xavier_normal_(self.bj.weight)

    def forward(self, i_indices, j_indices):
        w_i = self.wi(i_indices)
        w_j = self.wj(j_indices)
        b_i = self.bi(i_indices).squeeze()
        b_j = self.bj(j_indices).squeeze()

        x = torch.sum(w_i * w_j, dim=1) + b_i + b_j

        return x


if __name__ == "__main__":
    # glove = GloveModel(torch.Tensor(1), torch.Tensor(1))
    glove = GloveModel(100, 10)
    x = glove(torch.LongTensor([1, 2]), torch.LongTensor([1]))
    # x = glove(1, 1)
    print(x)
