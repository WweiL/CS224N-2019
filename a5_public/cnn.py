#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1i
import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F

class CNN(nn.Module):
    """ CNN
    """
    def __init__(self, num_channels, embed_char_size, kernel_size=5):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels = embed_char_size,
            out_channels = num_channels,
            kernel_size = kernel_size
        )
        self.relu = nn.ReLU()

    def forward(self, x_reshaped):
        """
        @param x_reshaped: shape (batch_size, e_char, m_word)
        @return: shape(batch_size, e_char)
        """
        x_conv_relu = self.relu(self.conv(x_reshaped))
        # assert x_conv_relu.shape == torch.Size([x_conv_relu.shape[0], 2, 8-5+1]), "Tensor shape is incorrect: it should be:\n {} but is:\n{}".format([x_conv_relu.shape[0], 2, 8-2+1], x_conv_relu.shape)
        return F.max_pool1d(x_conv_relu, kernel_size=x_conv_relu.shape[2]).squeeze(2)

### END YOUR CODE
