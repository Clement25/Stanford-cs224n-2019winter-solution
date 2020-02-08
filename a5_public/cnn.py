#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

### YOUR CODE HERE for part 1i
class CNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5):
        super(CNN,self).__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)

    def forward(self, x_reshaped):
        """Foraward passing of input tensor
        @param x_reshaped (Tensor): Input reshaped char embedding tensor with shape (batch_size, embed_size, max_seq_len)
        @return x_conv_out (Tensor): Word embeddings built on char embeddings with shape (batch_size, embed_size)
        """
        assert x_reshaped.size(-1) >= self.kernel_size
        x_conv = self.conv(x_reshaped) # shape of (batch_size, num_kernel, max_seq_len-kernel_size+1)
        x_conv_out = F.max_pool1d(F.relu(x_conv),kernel_size=x_conv.size(-1)).squeeze_(-1)
        return x_conv_out

### END YOUR CODE
if __name__ == '__main__':
    x = torch.rand(2,4,20)
    cnn = CNN(4, 8)
    print(cnn(x))
