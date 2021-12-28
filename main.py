#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author:   Luo Xukun
# Date:     2021.10.29

import loguru
import torch

from utils.config import *
from utils.data_preprocess import get_tag2id

from utils.data_loader import get_loader
from utils.tools import set_seed, load_parameters
from model.encoder import MyBertEncoder
from model.model import BiTTModel
from model.scripts import BiTTMetricsCalculator
from model.tagging_scheme import BidirectionalTreeTaggingScheme
from utils.joint_framework import JointExtractionFramework

def main():
    # Load parameters.
    args = load_parameters()

    # Log setting.
    if os.path.exists(args.log_path):
        os.remove(args.log_path)
    args.logger = loguru.logger
    args.logger.add(args.log_path)

    args.logger.info("seq_max_length: {}, data_name: {}, language: {}".format(args.seq_max_length, args.data_name, args.language))

    # Set seed.
    set_seed(args.seed)

    # Model.
    args.logger.info("Loading models...")
    args.tag2id = get_tag2id(args.data_name)
    args.tagger = BidirectionalTreeTaggingScheme(args.seq_max_length, len(args.tag2id.keys()))
    args.encoder = MyBertEncoder(model_type_dict[args.language])
    args.metrics_calculator = BiTTMetricsCalculator(args.tagger)
    args.model = BiTTModel(args)

    # Dataset.
    args.logger.info("Loading train dataset...")
    args.train_data_loader = get_loader(args, mode=0)
    args.logger.info("Loading valid dataset...")
    args.valid_data_loader = get_loader(args, mode=1)
    args.logger.info("Loading test dataset...")
    args.test_data_loader = get_loader(args, mode=2)

    # Framework
    args.logger.info("Building up Joint extraction framework...")
    args.framework = JointExtractionFramework(args)

    # Train, valid and test.
    if args.train:
        args.framework.train(args)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.framework.eval(args.model.to(device), args.metrics_calculator, device, args.load_ckpt)

if __name__ == "__main__":
    main()