from collections import namedtuple
import sys
from typing import List, Tuple, Dict, Set, Union
import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from model_embeddings import ModelEmbeddings

def pad_sents(sents, pad_token):
    """ Pad list of sentences according to the longest sentence in the batch.
    @param sents (list[list[str]]): list of sentences, where each sentence
                                    is represented as a list of words
    @param pad_token (str): padding token
    @returns sents_padded (list[list[str]]): list of sentences where sentences shorter
        than the max length sentence are padded out with the pad_token, such that
        each sentences in the batch now has equal length.
    """
    sents_padded = []

    ### YOUR CODE HERE (~6 Lines)
    max_len = len(max(sents, key=len))
    for sent in sents:
        pad_len = max_len - len(sent)
        sents_padded.append(sent+[pad_token]*pad_len)

    ### END YOUR CODE
    return sents_padded

# a = torch.FloatTensor([[2,1,1],[3,2,0]]).view(3,2,1)
# encoder = nn.LSTM(1,2,1,False,True)
# b,c = encoder(a)
# print(b)
# print(c)
a = torch.LongTensor(2,3)
print(a)