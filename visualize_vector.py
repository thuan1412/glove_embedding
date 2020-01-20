import torch
import torch.nn as nn

import json

from glove_model import GloveModel
from train import EMBED_DIM, device

from gensim.models import KeyedVectors

cos = nn.CosineSimilarity()
mapping_dicts = json.loads(open("mapping.json").read())

print(device)
glove_model = GloveModel(mapping_dicts['vocab_size'], EMBED_DIM)
glove_model.load_state_dict(torch.load("./text8.pt"))
glove_model.to(device)
glove_model.eval()

embedding = glove_model.wi.weight.cpu().data.numpy() + glove_model.wj.weight.cpu().data.numpy()
print(embedding.shape)


def words_similar(word1, word2):
    vector1 = embedding[mapping_dicts['word2id'][word1]]
    vector2 = embedding[mapping_dicts['word2id'][word2]]
    a = cos(torch.Tensor([vector1]).to(device), torch.Tensor([vector2]).to(device)).item()
    return a

def most_similar(word, n=10):
    vector = embedding[mapping_dicts['word2id'][word]]
    similar_vector = [ cos(torch.Tensor([vector]).to(device), torch.Tensor([i]).to(device)).item()
                        for i in embedding ]
    similar_dict = {word: vector for word, vector in zip(mapping_dicts['id2word'].values(), similar_vector)}
    similar_dict = sorted(similar_dict.items(), key= lambda item: item[1], reverse=True)
    return similar_dict[1:n+1]

def store_embedding_vector(filename="glove.txt"):
    f = open(filename, "w")
    f.write(f"{mapping_dicts['vocab_size']} {EMBED_DIM}\n")
    # f.write(f"10 {EMBED_DIM}\n")
    for i in range(mapping_dicts['vocab_size']):
        f.write(mapping_dicts['id2word'][str(i)] + ' ')
        string = str(embedding[i])
        string = string.replace('\n', ' ')
        string = string.replace('  ', ' ')
        string = string.replace('  ', ' ')
        string = string.replace('  ', ' ')
        string = string.replace('  ', ' ')

        f.write(string[2:-2])
        f.write('\n')
    f.close()

# while True:
#     x = input("Enter word: ")
#     print(most_similar(x))
# store_embedding_vector()

model = KeyedVectors.load_word2vec_format("glove.txt")