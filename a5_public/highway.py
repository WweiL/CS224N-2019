#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1h
import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F

class Highway(nn.Module):
    """
    Highway network
    """
    def __init__(self, embed_size, dropout_rate=0.5):
        """ init
        @param embed_size (int): Word Embedding size
        """
        super().__init__()
        self.embed_size = embed_size
        self.projection = nn.Linear(embed_size, embed_size)
        self.gate = nn.Linear(embed_size, embed_size)
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_conv_out: torch.Tensor) -> torch.Tensor:
        """ Compute x_highway from x_conv_out
        @param conv_out: tensor output from CNN, shape (batch, embed_size)
        @returns x_highway: result of applying the highway network to x_conv_out, shape(batch, embed_size)
        """
        x_proj = self.relu(self.projection(x_conv_out))
        x_gate = self.sigmoid(self.gate(x_conv_out))
        x_highway = x_gate * x_proj + (1 - x_gate) * x_proj
        return self.dropout(x_highway)

### END YOUR CODE
