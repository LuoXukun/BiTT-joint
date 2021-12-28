#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author:   Luo Xukun
# Date:     2021.12.20

import torch
import random
import numpy as np
import argparse

from utils.config import *

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def load_parameters():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Path options.
    parser.add_argument("--load_ckpt", default=None, type=str,
                        help="Directory of checkpoints for loading. Default: None.")
    parser.add_argument("--save_ckpt", default=None, type=str,
                        help="Directory of checkpoints for saving. Default: None.")
    parser.add_argument("--log_path", default=None, type=str,
                        help="Log path.")

    # Model options.
    parser.add_argument("--train", type=int, default=1,
                        help="If train or test. 1->train, 0->test.")

    # Training options.
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch_size.")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate.")
    parser.add_argument("--lstm_hidden", type=int, default=768,
                        help="Hidden size of the LSTM decoder.")
    parser.add_argument("--lstm_layers", type=int, default=2,
                        help="Layer number of the LSTM decoder.")
    parser.add_argument("--lstm_dropout", type=float, default=0.1,
                        help="Dropout rate of the LSTM decoder.")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate of other layers.")
    parser.add_argument("--train_epoch", type=int, default=20,
                        help="Num of epochs of training.")
    parser.add_argument("--val_epoch", type=int, default=2,
                        help="Validate every train val_step epochs.")
    parser.add_argument("--report_step", type=int, default=1000,
                        help="Validate every train train_step steps.")
    parser.add_argument("--warmup_rate", type=float, default=0.1,
                        help="Warming up rate of training.")
    parser.add_argument("--grad_iter", type=int, default=1,
                        help="Iter of gradient descent for training. Default: 1.")
    parser.add_argument('--seed', type=int, default=7,
                        help='random seed.')
    parser.add_argument("--use_fp16", type=int, default=1,
                        help="If use fp16.")
    
    # Data options.
    #parser.add_argument('--num_workers', type=int, default=4,
    #                    help='Number of thread workers for data loader.')
    parser.add_argument("--data_name", 
        choices=["NYT"],
        default="NYT",
        help="Name of dataset.")
    parser.add_argument("--language", 
        choices=["en", "ch"],
        default="en",
        help="The language.")
    
    args = parser.parse_args()

    # Others.
    args.seq_max_length = seq_max_length
    args.train = True if args.train else False
    args.use_fp16 = True if args.use_fp16 else False
    
    return args
