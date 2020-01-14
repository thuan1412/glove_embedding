import torch
import torch.nn as nn

import json

from glove_model import GloveModel
from train import EMBED_DIM, dataset

cos = nn.CosineSimilarity()

glove_model = GloveModel(dataset._vocab_len, EMBED_DIM)
glove_model.load_state_dict(torch.load("./text8.pt"))
glove_model.eval()

embedding = glove_model.wi.weight.data.numpy() + glove_model.wj.weight.data.numpy()
print(embedding.shape)

mapping_dicts = json.loads(open("mapping.json").read())

def words_similar(word1, word2):
    vector1 = embedding[mapping_dicts['word2id'][word1]]
    vector2 = embedding[mapping_dicts['word2id'][word2]]
    a = cos(torch.Tensor([vector1]), torch.Tensor([vector2])).item()
    return a

def most_similar(word, n=10):
    vector = embedding[mapping_dicts['word2id'][word]]
    similar_vector = [ cos(torch.Tensor([vector]), torch.Tensor([i])).item()
                        for i in embedding ]
    similar_dict = {word: vector for word, vector in zip(mapping_dicts['id2word'].values(), similar_vector)}
    similar_dict = sorted(similar_dict.items(), key= lambda item: item[1], reverse=True)
    return similar_dict[1:n+1]
most_similar("tsunami")