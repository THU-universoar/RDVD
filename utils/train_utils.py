import argparse
import os
import logging
import numpy as np
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from datetime import datetime
from torch.serialization import default_restore_location

import sys
sys.path.append('../')
import models

def setup_experiment(args):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.dry_run:
        args.no_save = args.no_log = args.no_visual = True
        return

    if not args.resume_training:
        args.experiment = "-".join([args.experiment, datetime.now().strftime("%b-%d-%H_%M_%S")])

    args.experiment_dir = os.path.join(args.output_dir, (f"drafts/" if args.draft else "") , args.experiment)
    os.makedirs(args.experiment_dir, exist_ok=True)

    if not args.no_save:
        args.checkpoint_dir = os.path.join(args.experiment_dir, "checkpoints")
        os.makedirs(args.checkpoint_dir, exist_ok=True)

    if not args.no_log:
        args.log_dir = os.path.join(args.experiment_dir, "logs")
        os.makedirs(args.log_dir, exist_ok=True)
        args.log_file = os.path.join(args.log_dir, "train.log")


def init_logging(args):
    handlers = [logging.StreamHandler()]
    if not args.no_log and args.log_file is not None:
        mode = "a" if os.path.isfile(args.resume_training) else "w"
        handlers.append(logging.FileHandler(args.log_file, mode=mode))
    logging.basicConfig(handlers=handlers, format="[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
    logging.info("COMMAND: %s" % " ".join(sys.argv))
    logging.info("Arguments: {}".format(vars(args)))


def save_checkpoint(args, step, model, optimizer=None, scheduler=None, score=None, mode="min"):
    assert mode == "min" or mode == "max"
    last_step = getattr(save_checkpoint, "last_step", -1)
    save_checkpoint.last_step = max(last_step, step)

    default_score = float("inf") if mode == "min" else float("-inf")
    best_score = getattr(save_checkpoint, "best_score", default_score)
    if (score < best_score and mode == "min") or (score > best_score and mode == "max"):
        save_checkpoint.best_step = step
        save_checkpoint.best_score = score

    if not args.no_save and step % args.save_interval == 0:
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        model = [model] if model is not None and not isinstance(model, list) else model
        optimizer = [optimizer] if optimizer is not None and not isinstance(optimizer, list) else optimizer
        scheduler = [scheduler] if scheduler is not None and not isinstance(scheduler, list) else scheduler
        state_dict = {
            "step": step,
            "score": score,
            "last_step": save_checkpoint.last_step,
            "best_step": save_checkpoint.best_step,
            "best_score": getattr(save_checkpoint, "best_score", None),
            "model": [m.state_dict() for m in model] if model is not None else None,
            "optimizer": [o.state_dict() for o in optimizer] if optimizer is not None else None,
            "scheduler": [s.state_dict() for s in scheduler] if scheduler is not None else None,
            "args": argparse.Namespace(**{k: v for k, v in vars(args).items() if not callable(v)}),
        }

        if args.step_checkpoints:
            torch.save(state_dict, os.path.join(args.checkpoint_dir, "checkpoint{}.pt".format(step)))
        if (score < best_score and mode == "min") or (score > best_score and mode == "max"):
            torch.save(state_dict, os.path.join(args.checkpoint_dir, "checkpoint_best.pt"))
        if step > last_step:
            torch.save(state_dict, os.path.join(args.checkpoint_dir, "checkpoint_last.pt"))


def load_checkpoint(args, model=None, optimizer=None, scheduler=None):
    if args.restore_file is not None and os.path.isfile(args.restore_file):
        print('restoring model..')
        state_dict = torch.load(args.restore_file, map_location=lambda s, l: default_restore_location(s, "cpu"))

        model = [model] if model is not None and not isinstance(model, list) else model
        optimizer = [optimizer] if optimizer is not None and not isinstance(optimizer, list) else optimizer
        scheduler = [scheduler] if scheduler is not None and not isinstance(scheduler, list) else scheduler

        if "best_score" in state_dict:
            save_checkpoint.best_score = state_dict["best_score"]
            save_checkpoint.best_step = state_dict["best_step"]
        if "last_step" in state_dict:
            save_checkpoint.last_step = state_dict["last_step"]
        if model is not None and state_dict.get("model", None) is not None:
            for m, state in zip(model, state_dict["model"]):
                m.load_state_dict(state)
        if optimizer is not None and state_dict.get("optimizer", None) is not None:
            for o, state in zip(optimizer, state_dict["optimizer"]):
                o.load_state_dict(state)
        if scheduler is not None and state_dict.get("scheduler", None) is not None:
            for s, state in zip(scheduler, state_dict["scheduler"]):
                milestones = s.milestones
                state['milestones'] = milestones
                s.load_state_dict(state)
                s.milestones = milestones

        logging.info("Loaded checkpoint {}".format(args.restore_file))
        return state_dict
    
# definition for loading model from a pretrained network file
def load_model(PATH, Fast=False, parallel=False, pretrained=True, old=True, load_opt=False, mf2f=False):
    if not Fast:
        state_dict = torch.load(PATH, map_location="cpu")
        args = argparse.Namespace(**{**vars(state_dict["args"])})
        # ignore this
        if old:
            vars(args)['blind_noise'] = False

        model = models.build_model(args)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    else:
        model = models.FastDVDnet(mf2f=mf2f)
    
    if load_opt:
        for o, state in zip([optimizer], state_dict["optimizer"]):
            o.load_state_dict(state)
    
    if pretrained:
        if Fast:
            state_dict = torch.load(PATH)
        else:
            state_dict = torch.load(PATH)["model"][0]
        own_state = model.state_dict()
        
        for name, param in state_dict.items():
            if parallel:
                name = name[7:]
            if Fast and not mf2f:
                name = name.split('.', 1)[1]
            if name not in own_state:
                print("not matching: ", name)
                continue
            if isinstance(param, nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            own_state[name].copy_(param)
        
    if not Fast:
        return model, optimizer, args
    else:
        return model
