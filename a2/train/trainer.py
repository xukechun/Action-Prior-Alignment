import os
import time
import datetime
import random
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from helpers.logger import Logger
from helpers.data_loader import unified_data_loader, unified_adaptive_data_loader
from models.networks import CLIPAction, AdaptPolicyCLIPAction, AdaptFeatCLIPAction, CLIPLangEmbAction
from tensorboardX import SummaryWriter

from env.constants import WORKSPACE_LIMITS
import utils.utils as utils

class BaseTrainer:
    def __init__(self, model, args):
        self.model = model
        self.args = args
        self.device = args.device
        self.iteration = 0
        self.current_epoch = 0
        
        # Setup optimizer
        self.optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=2e-5, betas=(0.9,0.99))
        if args.adjust_lr:
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, 
                step_size=args.step_size, 
                gamma=args.step_ratio
            )

    def train_epoch(self, train_dl, recorder):
        self.model.train()
        losses = []
        sample_num = 0
        
        for i, batch in enumerate(train_dl):
            loss = self.train_step(batch, recorder)
            if loss is not None:
                losses.append(loss)
                sample_num += 1
        
        avg_loss = sum(losses) / len(losses) if losses else 0
        print(f"\033[034m Epoch: {self.current_epoch}, sample: {sample_num}, loss: {avg_loss}\033[0m")
        recorder['tb'].add_scalar('loss/epoch', avg_loss, global_step=self.current_epoch)
        
        if (self.current_epoch + 1) % 5 == 0:
            recorder['logger'].save_sl_checkpoint(
                self.model, 
                datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), 
                str(self.current_epoch)
            )

    def train_step(self, batch, recorder):
        raise NotImplementedError("Subclasses must implement train_step")

    def train(self, train_dl, recorder, start_epoch=0):
        self.current_epoch = start_epoch
        for epoch in range(start_epoch, start_epoch + self.args.epoch_num):
            self.train_epoch(train_dl, recorder)
            if self.args.adjust_lr:
                self.scheduler.step()
            self.current_epoch += 1

class Trainer(BaseTrainer):
    def __init__(self, model, args):
        super().__init__(model, args)
        # Setup loss function based on args
        self.create_criterion()

    def create_criterion(self):
        if self.args.loss == "ce":
            self.criterion = nn.CrossEntropyLoss()
        elif self.args.loss == "mse":
            self.criterion = nn.MSELoss()
        elif self.args.loss == "nll":
            self.criterion = nn.NLLLoss()
        elif self.args.adaptive:
            self.criterion = nn.BCEWithLogitsLoss()

    def train_step(self, batch, recorder):
        if self.args.adaptive:
            sequence, lang_goal, pts_pos, pts_feat, pts_sim, actions, gt_action_idxs = batch
        else:
            sequence, lang_goal, pts_pos, pts_feat, pts_sim, actions, action_idx, done = batch
            if not done:
                return None

        # Move data to device
        pts_pos = pts_pos.to(self.device)
        pts_feat = pts_feat.to(self.device)
        pts_sim = pts_sim.to(self.device)
        actions = actions.to(self.device)
        lang_goal = lang_goal[0]

        # Normalize if needed
        if self.args.normalize:
            pts_pos = utils.normalize_pos(pts_pos, WORKSPACE_LIMITS.T, device=pts_pos.device)
            actions[:, :, :3] = utils.normalize_pos(actions[:, :, :3], WORKSPACE_LIMITS.T, device=pts_pos.device)

        # Determine task mode
        if self.args.task_emb:
            mode = "grasp"
            place_verbs = ["put", "place", "move"]
            for verb in place_verbs:
                if verb in lang_goal:
                    mode = "place"
        else:
            mode = None

        # Forward pass
        if self.args.adaptive:
            gt_action_logits = torch.zeros(actions.shape[1])
            gt_action_logits[gt_action_idxs] = 1
            gt_action_logits = gt_action_logits.unsqueeze(0).to(self.device)
            pred_action_logits, _ = self.model(pts_pos, pts_feat, pts_sim, actions, mode)
            loss = self.criterion(pred_action_logits, gt_action_logits)
        else:
            action_idx = torch.from_numpy(np.array(action_idx)).to(self.device)

            if not self.args.lang_emb:
                pred_action_logits, _ = self.model(pts_pos, pts_feat, pts_sim, actions, mode)
            else:
                pred_action_logits, _ = self.model(pts_pos, pts_feat, actions, lang_goal, mode)
            
            if self.args.loss == "ce":
                loss = self.criterion(pred_action_logits, action_idx)
            elif self.args.loss == "mse":
                pred_action_logits_softmax = F.softmax(pred_action_logits, dim=-1)
                pred_logits = pred_action_logits_softmax[0][action_idx.item()].unsqueeze(0)
                loss = self.criterion(pred_logits, torch.ones(1).to(self.device))
            elif self.args.loss == "nll":
                pred_action_logits_sigmoid = F.logsigmoid(pred_action_logits)
                loss = self.criterion(pred_action_logits_sigmoid, action_idx)

        loss = loss.sum()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        print(f"\033[034m Epoch: {self.current_epoch}, iteration: {self.iteration}, sequence: {sequence[0]}, lang goal: {lang_goal}, loss: {loss}\033[0m")
        recorder['tb'].add_scalar('loss/iteration', loss, global_step=self.iteration)
        self.iteration += 1
        
        return loss
