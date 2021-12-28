#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author:   Luo Xukun
# Date:     2021.10.21

import os
import torch
import torch.nn as nn

from torch.nn import DataParallel
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup

class JointExtractionFramework(object):
    def __init__(self, args):
        super(JointExtractionFramework, self).__init__()
        self.train_data_loader = args.train_data_loader
        self.valid_data_loader = args.valid_data_loader
        self.test_data_loader = args.test_data_loader
        self.logger = args.logger

    def __load_model__(self, ckpt):
        if os.path.isfile(ckpt):
            checkpoint = torch.load(ckpt)
            self.logger.info("Successfully loaded checkpoint '{}'".format(ckpt))
            return checkpoint
        else:
            self.logger.critical("No checkpoint found at '{}'".format(ckpt))
            exit()
    
    def train(self, args):
        """ 
            Training step.

            Args:

                model:                  BiTT model.
                metrics_calculator:     Few shot metrics calculator.
                batch_size:             Batch size.
                learning_rate:          Initial train learning rate.
                train_epoch:            Num of epochs of training.
                val_epoch:              Validate every train val_step epochs.
                report_step:            Validate every train train_step steps.
                load_ckpt:              Directory of checkpoints for loading. Default: None.
                save_ckpt:              Directory of checkpoints for saving. Default: None.
                warmup_rate:            Warming up rate of training.
                grad_iter:              Iter of gradient descent. Default: 1.
                tag_seqs_num:           Tag seqs number.
                use_fp16:               If use fp16.
            
            Returns:
        """
        self.logger.info("Start Training...")

        # Load model.
        if args.load_ckpt:
            self.logger.info("Loading checkpoint '{}'...".format(args.load_ckpt))
            state_dict = self.__load_model__(args.load_ckpt)['state_dict']
            own_state = args.model.state_dict()
            for name, param in state_dict.items():
                if name not in own_state:
                    self.logger.warning("Ignore {}".format(name))
                    continue
                self.logger.info("Loading {} from {}".format(name, args.load_ckpt))
                own_state[name].copy_(param)
        
        # For simplicity, we use DataParallel wrapper to use multiple GPUs.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count() > 1:
            self.logger.info("{} GPUs are available. Let's use them.".format(torch.cuda.device_count()))
            args.model = nn.DataParallel(args.model)
        args.model = args.model.to(device)
        args.loss = args.model.module.loss if torch.cuda.device_count() > 1 else args.model.loss

        # Init optimizer.
        self.logger.info("Use Bert Optim...")
        parameters_to_optimize = list(args.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        parameters_to_optimize = [
            {'params': [p for n, p in parameters_to_optimize if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in parameters_to_optimize if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(parameters_to_optimize, lr=args.learning_rate, correct_bias=False)

        if args.use_fp16:
            from apex import amp
            args.model, optimizer = amp.initialize(args.model, optimizer, opt_level="O1")
        
        total_steps = int(args.train_epoch * len(self.train_data_loader.dataset) / args.batch_size) + 1
        warmup_steps = int(total_steps * args.warmup_rate)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

        # Start Training.
        best_f1 = 0.0

        for epoch_id in range(1, args.train_epoch + 1):
            iter_loss, iter_accs = 0.0, None
            args.model.train()

            # Train.
            for batch_id, batch in enumerate(self.train_data_loader):
                for k in batch:
                    if k != "sample":
                        batch[k] = batch[k].to(device)
                
                #print("src: {}, tags: {}".format(batch["src_ids"].shape, batch["tags"].shape))
                logits, preds = args.model(batch)
                preds = torch.stack(preds, 0)
                #print("logits: {}, preds: {}".format(logits[0].shape, preds.shape))

                if (batch_id + 1) % args.report_step == 0 or batch_id == len(self.train_data_loader.dataset) - 1:
                    loss = args.loss(logits, batch["tags"], quiet=False) / float(args.grad_iter)
                else:
                    loss = args.loss(logits, batch["tags"]) / float(args.grad_iter)
                #accs = args.metrics_calculator.get_accs(preds, batch["tags"])
                iter_loss += loss.data.item()

                if args.use_fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                
                if batch_id % args.grad_iter == 0 or batch_id == len(self.train_data_loader.dataset) - 1:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                """ if not iter_accs:
                    iter_accs = {key: 0.0 for key, value in accs.items()}
                for key, value in accs.items():
                    iter_accs[key] += value """
                
                if (batch_id + 1) % args.report_step == 0 or batch_id == len(self.train_data_loader.dataset) - 1:
                    batch_format = "TRAIN -- Epoch: {}, Step: {}, loss: {},".format(epoch_id, batch_id + 1, iter_loss / float(args.report_step))
                    """ for key, value in iter_accs.items():
                        batch_format += " {}: {},".format(key, value / float(args.report_step)) """
                    batch_format = batch_format[:-1]
                    self.logger.info(batch_format)
                    iter_loss = 0.0; iter_accs = None
            
            # Valid.
            if epoch_id % args.val_epoch == 0 or epoch_id == args.train_epoch:
                val_f1 = self.eval(args.model, args.metrics_calculator, device)
                if val_f1 > best_f1 or epoch_id == args.val_epoch:
                    if args.save_ckpt:
                        self.logger.info("Better checkpoint! Saving...")
                        state_dict = args.model.module.state_dict() if torch.cuda.device_count() > 1 else args.model.state_dict()
                        torch.save({"state_dict": state_dict}, args.save_ckpt)
                    best_f1 = val_f1
        
        # Testing.
        self.logger.info("Start Testing...")
        if args.save_ckpt:
            test_f1 = self.eval(args.model, args.metrics_calculator, device, ckpt=args.save_ckpt)
        else:
            self.logger.warning("There is no a saved checkpoint path, so we cannnot test on the best model!")
        
        self.logger.info("Finish Training")
    
    def eval(self, model, metrics_calculator, device, ckpt=None):
        '''
            Validation.
            Args:
                model:                  FewShotREModel instance.
                metrics_calculator:     Few shot metrics calculator.
                device:                 CPU or GPU.
                ckpt:                   Checkpoint path. Set as None if using current model parameters.
            Returns: 
                f1
        '''
        self.logger.info("Evaluation...")
        model.eval()

        eval_data_loader = self.test_data_loader if ckpt else self.valid_data_loader
        if ckpt is not None:
            self.logger.info("Loading best checkpoint '{}'...".format(ckpt))
            state_dict = self.__load_model__(ckpt)['state_dict']
            own_state = model.state_dict()
            for name, param in state_dict.items():
                if name not in own_state:
                    continue
                own_state[name].copy_(param)
        
        pred_cnt, gold_cnt, correct_cnt = 0, 0, 0

        with torch.no_grad():
            for batch_id, batch in enumerate(eval_data_loader):
                #print("batch_id: ", batch_id)
                for k in batch:
                    if k != "sample":
                        batch[k] = batch[k].to(device)

                logits, preds = model(batch)
                preds = torch.stack(preds, 0)

                tmp_pred_cnt, tmp_gold_cnt, tmp_correct_cnt = metrics_calculator.get_rel_pgc(batch["sample"], preds)
                pred_cnt += tmp_pred_cnt; gold_cnt += tmp_gold_cnt; correct_cnt += tmp_correct_cnt
            
            prec, rec, f1 = self.get_prf(pred_cnt, gold_cnt, correct_cnt)
        
        self.logger.critical(
            "TRAIN EVAL -- Eval_instances: {}, Pred: {}, Gold: {}, Correct: {}, Prec: {}, Rec: {}, F1: {}".format(
                len(eval_data_loader.dataset), pred_cnt, gold_cnt, correct_cnt, prec, rec, f1
            )
        )

        return f1
    
    def get_prf(self, pred, gold, correct):
        mini_num = 1e-10
        precision = correct / (pred + mini_num)
        recall = correct / (gold + mini_num)
        f1 = 2 * precision * recall / (precision + recall + mini_num)
        return precision, recall, f1