#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author:   Luo Xukun
# Date:     2021.12.16

""" Bidirectional Tree Tagging Model. """

import torch
import torch.nn as nn
import numpy as np

class BiTTModel(nn.Module):
    def __init__(self, args):
        super(BiTTModel, self).__init__()
        self.encoder = args.encoder
        self.hidden_size = self.encoder.config.hidden_size
        self.lstm_hidden = args.lstm_hidden
        self.lstm_layers = args.lstm_layers
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.parts_num = args.tagger.parts_num
        self.tags_num = args.tagger.tags_num
        self.tags_weight = args.tagger.tags_weight
        self.parts_weight = args.tagger.parts_weight
        self.categories_num = args.tagger.categories_num

        # Bi-LSTM layer.
        self.lstm = nn.LSTM(
            input_size = self.hidden_size,
            hidden_size = self.lstm_hidden,
            num_layers = self.lstm_layers,
            bidirectional = True,
            dropout = args.lstm_dropout,
            batch_first = True
        )

        # Linear layers.
        self.linear = nn.ModuleList([
            nn.Linear(self.lstm_hidden * 2, self.tags_num[i] * self.categories_num * 2) for i in range(self.parts_num)
        ])

        # Dropout layer.
        self.drop = nn.Dropout(p=args.dropout)

        # Cost.
        self.cost = [nn.CrossEntropyLoss(weight=torch.Tensor(self.tags_weight[i])).to(self.device) for i in range(self.parts_num)]
    
    def forward(self, batch):
        batch_size, seq_len = batch["src_ids"].size(0), batch["src_ids"].size(1) - 1

        # Embedding. (batch_size, seq_len, hidden_size)
        embedding = self.encoder(batch["src_ids"], batch["mask_ids"], batch["seg_ids"])[:, 1:, :]
        embedding = self.drop(embedding)

        # Hidden. (batch_size * seq_len, lstm_hidden * 2)
        lstm_h = self.__init_hidden__(batch_size, self.device)
        hidden, lstm_h = self.lstm(embedding, lstm_h)
        hidden = hidden.contiguous().view(-1, self.lstm_hidden * 2)
        hidden = self.drop(hidden)

        # Forward and Backward Feats. (batch_size, categories_num, seq_len, tags_num)
        forward_feats, backward_feats = [], []
        forward_preds, backward_preds = [], []
        for i in range(self.parts_num):
            feats = self.linear[i](hidden).view(batch_size, seq_len, 2, self.categories_num, -1).permute(2, 0, 3, 1, 4)
            forward_feats.append(feats[0])
            backward_feats.append(feats[1])
            forward_preds.append(torch.max(feats[0], -1)[1])
            backward_preds.append(torch.max(feats[1], -1)[1])
        
        # Return. (2 * parts_num, batch_size, categories_num, seq_len, tags_num), (2 * parts_num, batch_size, categories_num, seq_len)
        return forward_feats + backward_feats, forward_preds + backward_preds
    
    def __init_hidden__(self, batch_size, device):
        return (torch.zeros(2 * self.lstm_layers, batch_size, self.lstm_hidden, device=self.device),
                torch.zeros(2 * self.lstm_layers, batch_size, self.lstm_hidden, device=self.device))
    
    def loss(self, logits, label, quiet=True):
        '''
            Args:

                logits:     Logits with the size (2 * parts_num, batch_size, categories_num, seq_len, tags_num).
                label:      Label with the size (2 * parts_num, batch_size, categories_num, seq_len).

            Returns: 

                [Loss] (A single value)
        '''
        loss = []
        for i in range(self.parts_num * 2):
            N = logits[i].size(-1)
            loss.append(self.cost[i % self.parts_num](logits[i].contiguous().view(-1, N), label[i].contiguous().view(-1)))
        
        if not quiet:
            str_format = "Loss of every part: "
            for i in range(self.parts_num * 2):
                str_format += "{}, ".format(loss[i])
            print(str_format[:-2])

        total_loss = 0.0
        for i in range(self.parts_num * 2):
            total_loss += (self.parts_weight[i % self.parts_num] * loss[i])
        return total_loss