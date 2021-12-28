#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author:   Luo Xukun
# Date:     2021.12.20

import os
root_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

data_path_temp = [
    os.path.join(root_path, "datasets/<name_template>/train.json"),
    os.path.join(root_path, "datasets/<name_template>/valid.json"),
    os.path.join(root_path, "datasets/<name_template>/test.json")
]

model_type_dict = {"en": "bert-base-cased", "ch": "bert-base-chinese"}

seq_max_length = 128