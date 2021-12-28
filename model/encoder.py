#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author:   Luo Xukun
# Date:     2021.12.20

import torch
import torch.nn as nn

from transformers import BertModel

class MyBertEncoder(nn.Module):
    def __init__(self, model_type_dict):
        super(MyBertEncoder, self).__init__()
        self.bert = BertModel.from_pretrained(model_type_dict)
        self.config = self.bert.config
    
    def forward(self, src_ids, mask_ids, seg_ids):
        outputs = self.bert(src_ids, mask_ids, seg_ids, output_hidden_states=True, return_dict=True)
        # use the sum of the last 4 layers
        last_four_hidden_states = torch.cat([hidden_state.unsqueeze(0) for hidden_state in outputs['hidden_states'][-4:]], 0)
        del outputs
        embeddings = torch.sum(last_four_hidden_states, 0) # [num_sent, number_of_tokens, 768]
        return embeddings