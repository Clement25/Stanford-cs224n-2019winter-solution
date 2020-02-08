#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

### YOUR CODE HERE for part 1h
class Highway(nn.Module):
    def __init__(self, embed_size):
        """ Init Highway Network

        @param embed_size (int): Embedding size (dimensionality) 
        @param drop_out (float): Dropout ratio for dropout layer
        """
        super(Highway, self).__init__()
        self.embed_size = embed_size
        self.linear_proj = nn.Linear(embed_size, embed_size, bias=True)
        self.linear_gate = nn.Linear(embed_size, embed_size, bias=True)

        # Only for test here

        """
        torch.nn.init.constant_(self.linear_proj.weight,1)
        torch.nn.init.constant_(self.linear_proj.bias,0.5)
        torch.nn.init.constant_(self.linear_gate.weight,0.5)
        torch.nn.init.constant_(self.linear_gate.bias,0.5) 
        """

    def forward(self, x_conv_out):
        """ Forward propogation

        @param embed x_conv_out (Tensor): Tensor of convnet output with shape (batch_size, embed_size)

        @returns x_word_emb (Tensor): Tensor of input word embeddings with shape (batch_size, embed_size)
        """
        x_proj = F.relu(self.linear_proj(x_conv_out))
        x_gate = F.softmax(self.linear_gate(x_conv_out),dim=1)
        x_highway = torch.mul(x_proj,x_gate)+torch.mul((1-x_gate),x_conv_out)
        return x_highway

### END YOUR CODE 

if __name__ == '__main__':
    hw = Highway(2,0)
    x_conv_out = torch.Tensor([0.5,0.5])
