#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author:   Luo Xukun
# Date:     2021.12.20

""" Data loader for BiTT model. """

# For test.
#import sys
#sys.path.append("../")

import os
import json
import copy
import torch
import loguru
import argparse

from tqdm import tqdm
from transformers import BertTokenizerFast
from torch.utils.data import Dataset, DataLoader

from model.scripts import BiTTDataMaker
from model.tagging_scheme import BidirectionalTreeTaggingScheme
from utils.data_preprocess import get_norm_data, get_tag2id
from utils.config import data_path_temp, model_type_dict, seq_max_length

class BiTTDataset(Dataset):
    def __init__(self, args, mode=0):
        """ 
            Joint Extraction Dataset.
            Args:
                args:       The arguments from command.
                mode:       The mode of data. 0 -> train, 1 -> valid, 2 -> test.
            Returns:
        """
        super(BiTTDataset, self).__init__()
        self.mode = mode
        self.data_name = args.data_name         # Name of dataset. Such as "NYT".
        self.language = args.language           # The language. "en" or "ch".
        self.logger = args.logger
        self.tag2id = args.tag2id
        self.id2tag = {index: tag for tag, index in self.tag2id.items()}
        self.data_path = data_path_temp[self.mode].replace("<name_template>", self.data_name)
        self.tokenizer = BertTokenizerFast.from_pretrained(model_type_dict[self.language])
        self.samples = self.__load_data__()
        self.length = len(self.samples)
        self.tagger = args.tagger
        self.seq_max_len = seq_max_length
        self.datamaker = BiTTDataMaker(self.tokenizer, self.tagger)

    def __load_data__(self):
        datas = []
        self.logger.info("Loading data from {}".format(self.data_path))
        with open(self.data_path, "r", encoding="utf-8") as f:
            for index, line in enumerate(f):
                datas.append(json.loads(line.strip()))
        new_datas = get_norm_data(datas, self.tokenizer, self.data_name, self.tag2id)
        return new_datas
    
    def __getitem__(self, index):
        index_data = self.datamaker.get_indexed_data(self.samples[index], self.mode, self.seq_max_len)
        dataset = {
            "src_ids": torch.LongTensor(index_data[0]),
            "seg_ids": torch.LongTensor(index_data[1]),
            "mask_ids": torch.LongTensor(index_data[2]),
            "tags": index_data[3],
            "sample": index_data[4]
        }
        return dataset

    def __len__(self):
        return self.length
    
def collate_fn(batch):
    new_batch = {"src_ids": [], "seg_ids": [], "mask_ids": [], "tags": [], "sample": []}
    for i in range(len(batch)):
        for k in new_batch.keys():
            new_batch[k].append(batch[i][k])
    for k in new_batch.keys():
        if k != "sample":
            new_batch[k] = torch.stack(new_batch[k], 0)
    # tags. (parts_num * 2, batch_size, categories_num, seq_len)
    new_batch["tags"] = new_batch["tags"].permute(2, 0, 1, 3)
    return new_batch

def get_loader(args, mode=0):
    """ 
        Get data loader for Joint Extration dataset.
        Args:
            args:       The arguments from command.
            mode:       The mode of data. 0 -> train, 1 -> valid, 2 -> test.
        Returns:
            data_loader
    """
    dataset = BiTTDataset(args, mode)
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    return data_loader

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    args = parser.parse_args()

    args.data_name = "NYT"                  # Name of dataset. Such as "NYT".
    args.language = "en"                    # The language. "en" or "ch".
    args.batch_size = 16
    #args.num_workers = 4
    args.seq_max_length = seq_max_length
    args.logger = loguru.logger
    args.tag2id = get_tag2id(args.data_name)
    args.tagger = BidirectionalTreeTaggingScheme(args.seq_max_length, len(args.tag2id.keys()))
    
    data_loader = get_loader(args, mode=1)
    
    for index, batch in enumerate(data_loader):
        if index < 1:
            print(batch["src_ids"].shape)
            print(batch["seg_ids"].shape)
            print(batch["mask_ids"].shape)
            print(batch["tags"].shape)
        else:
            break