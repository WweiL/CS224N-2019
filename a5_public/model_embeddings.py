#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
model_embeddings.py: Embeddings for the NMT model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Anand Dhoot <anandd@stanford.edu>
Michael Hahn <mhahn2@stanford.edu>
"""

import torch.nn as nn
import torch

# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(j)

from cnn import CNN
from highway import Highway

# End "do not change"

class ModelEmbeddings(nn.Module):
    """
    Class that converts input words to their CNN-based embeddings.
    """
    def __init__(self, embed_size, vocab_entry):
        """
        Init the Embedding layer for one language
        @param embed_size (int): Embedding size (dimensionality) for the output
        @param vocab_entry (VocabEntry): VocabEntry object. See vocab.py for documentation.
        """
        super(ModelEmbeddings, self).__init__()

        ## A4 code
        # pad_token_idx = vocab.src['<pad>']
        # self.embeddings = nn.Embedding(len(vocab.src), embed_size, padding_idx=pad_token_idx)
        ## End A4 code

        ### YOUR CODE HERE for part 1j
        self.dropout_rate = 0.3
        self.e_char = 50
        self.e_word = embed_size
        self.embed_size = embed_size
        
        self.dropout = nn.Dropout(self.dropout_rate)
        self.cnn = CNN(self.e_word, self.e_char)
        self.highway = Highway(self.e_word)

        self.vocab_entry = vocab_entry
        self.char_embeddings = nn.Embedding(len(vocab_entry.char2id), self.e_char, padding_idx=vocab_entry.char2id['<pad>'])
        ### END YOUR CODE

    def forward(self, input):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, embed_size), containing the
            CNN-based embeddings for each word of the sentences in the batch
        """
        ### YOUR CODE HERE for part 1j
        x_emb = self.char_embeddings(input)
        assert x_emb.shape == torch.Size([input.shape[0], input.shape[1], input.shape[2], self.e_char])
        x_reshaped = x_emb.permute(0, 1, 3, 2)
        x_conv_out = [self.cnn(x) for x in x_reshaped]
        x_highway = [self.highway(x) for x in x_conv_out]
        x_word_emb = [self.dropout(x) for x in x_highway]
        return torch.stack(x_word_emb)
        ### END YOUR CODE

        ## A4 code
        # output = self.embeddings(input)
        # return output
        ## End A4 code
