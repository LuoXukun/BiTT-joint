#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author:   Luo Xukun
# Date:     2021.12.16

""" BiTT datamaker and metrics. """

import torch

from model.tagging_scheme import BidirectionalTreeTaggingScheme

class BiTTDataMaker():
    def __init__(self, tokenizer, tagger: BidirectionalTreeTaggingScheme):
        """ DataMaker for Bidirectional Tree Tagging model. """
        self.tokenizer = tokenizer
        self.tagger = tagger
    
    def get_indexed_data(self, data, mode, seq_max_len):
        """ 
            Get the indexed data.
            Args:
                data:               {
                                        "tokens": ["a", "b", "c", ...],
                                        "relations": [
                                            {
                                                "label": "",
                                                "label_id": number,
                                                "subject": ["a"],
                                                "object": ["c"],
                                                "sub_span": (0, 1),
                                                "obj_span": (2, 3)
                                            }, ...
                                        ]
                                    }
                mode:               The mode of data. 0 -> train, 1 -> valid.
                seq_max_len:        Max length of the src_ids.
            Returns:
                (src_ids, seg_ids, mask_ids, tags, sample)
        """
        #print("Get indexed data...")
        tokens, relations = data["tokens"], data["relations"]
    
        # The tag_len is seq_max_len, and the src_len is seq_max_len + 1.
        # Since we add [CLS] at the begining of the sentence.
        src_res = self.tokenizer(tokens, is_split_into_words=True, truncation=True, max_length=seq_max_len+1, padding="max_length")
        src_ids, seg_ids, mask_ids = src_res["input_ids"], src_res["token_type_ids"], src_res["attention_mask"]

        # Tagging.
        sample = {
            "tokens": tokens,
            "relations": [relation for relation in relations if relation["sub_span"][1] <= seq_max_len and relation["obj_span"][1] <= seq_max_len]
        }
        tags = self.tagger.encode_rel_to_bitt_tag(sample)
        
        return src_ids, seg_ids, mask_ids, tags, sample

class BiTTMetricsCalculator():
    def __init__(self, tagger: BidirectionalTreeTaggingScheme):
        self.tagger = tagger
        self.parts_num = tagger.parts_num
        self.accs_keys = ["F-part{}".format(i + 1) for i in range(self.parts_num)] + \
            ["B-part{}".format(i + 1) for i in range(self.parts_num)]
    
    def get_accs(self, preds, labels):
        '''
            The accuracy of all pred labels of a sample are right.

            Args:

                preds:      Preds with the size (parts_num * 2, batch_size, categories_num, seq_len).
                label:      Label with the size (parts_num * 2, batch_size, categories_num, seq_len).

            Returns: 

                [accs] (A dict)
        '''
        accs, batch_size = {}, preds.size(1)

        for index, key in enumerate(self.accs_keys):
            pred_ids, label_ids = preds[index].contiguous().view(batch_size, -1), labels[index].contiguous().view(batch_size, -1)

            correct_tag_num = torch.sum(torch.eq(label_ids, pred_ids).float(), dim=1)
            
            # correct_tag_num == categories_num * seq_len.
            sample_acc_ = torch.eq(correct_tag_num, torch.ones_like(correct_tag_num) * label_ids.size()[-1]).float()
            sample_acc = torch.mean(sample_acc_)
            
            accs[self.accs_keys[index]] = sample_acc
        
        return accs
    
    def get_rel_pgc(self, samples, preds):
        '''
            Get pred, gold and correct.

            Args:

                samples:    Sample dict.
                            {
                                "tokens": ["a", "b", "c", ...],
                                "relations": [
                                    {
                                        "label": "",
                                        "label_id": number,
                                        "subject": ["a"],
                                        "object": ["c"],
                                        "sub_span": (0, 1),
                                        "obj_span": (2, 3)
                                    }, ...
                                ]
                            }
                preds:      Preds with the size (parts_num * 2, batch_size, categories_num, seq_len).

            Returns: 

                (pred, gold, correct)
        '''
        seq_len = preds.size(-1)
        #print("seq_len: ", seq_len)
        correct_num, pred_num, gold_num = 0, 0, 0

        for index in range(len(samples)):
            sample = samples[index]
            tokens = sample["tokens"]
            tags = preds[:, index, :, :]

            pred_rel_list = self.tagger.decode_rel_fr_bitt_tag(tokens, tags)

            gold_rel_list = sample["relations"]

            pred_rel_set = set([
                "{}, {}, {}, {}, {}".format(
                    rel["sub_span"][0], rel["sub_span"][1], rel["obj_span"][0], rel["obj_span"][1], rel["label_id"]
                ) for rel in pred_rel_list
            ])
            gold_rel_set = set([
                "{}, {}, {}, {}, {}".format(
                    rel["sub_span"][0], rel["sub_span"][1], rel["obj_span"][0], rel["obj_span"][1], rel["label_id"]
                ) for rel in gold_rel_list if rel["sub_span"][1] <= seq_len and rel["obj_span"][1] <= seq_len
            ])

            correct_num += len(pred_rel_set.intersection(gold_rel_set))
            pred_num += len(pred_rel_set)
            gold_num += len(gold_rel_set)
        
        return pred_num, gold_num, correct_num