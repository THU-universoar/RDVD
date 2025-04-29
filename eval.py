import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

import utils
from models.udvd import BlindVideoNet
from datasets import load_SingleVideo


def main(args):
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    # Build data loaders, a model and an optimizer
    model = BlindVideoNet(channels_per_frame=args.channels, out_channels=args.out_channels, bias=args.bias).to(device)
    criterion = nn.MSELoss(reduction='sum')
    criterion.to(device)
    cpf = model.c  # channels per frame
    mid = args.n_frames // 2
    model = nn.DataParallel(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1000], gamma=0.5)
    logging.info(f"Built a model consisting of {sum(p.numel() for p in model.parameters()):,} parameters")

    state_dict = utils.load_checkpoint(args, model, optimizer, scheduler)

    train_loader, valid_loader = load_SingleVideo(args.data_path, noisy_path=args.noisy_path,
                                                  batch_size=args.batch_size, image_size=args.image_size,
                                                  stride=args.stride, n_frames=args.n_frames, aug=args.aug)

    # Track moving average of loss values
    valid_meters = {name: utils.AverageMeter() for name in (["valid_loss", "valid_psnr"])}
    ########验证

    #######设置存储路径
    output_dir_epoch = os.path.join("valid")
    os.makedirs(output_dir_epoch, exist_ok=True)
    #######

    model.eval()
    for meter in valid_meters.values():
        meter.reset()

    valid_bar = utils.ProgressBar(valid_loader)
    running_valid_psnr = 0.0
    for sample_id, (sample, noisy_inputs) in enumerate(valid_bar):
        with (torch.no_grad()):
            sample = sample.to(device)
            noisy_inputs = noisy_inputs.to(device)
            outputs = model(noisy_inputs)
            noisy_frame = noisy_inputs[:, (mid * cpf):((mid + 1) * cpf), :, :]
            truth_frame = sample[:, (mid * cpf):((mid + 1) * cpf), :, :]

            loss = criterion(outputs, truth_frame)

            valid_psnr = utils.psnr(sample[:, (mid * cpf):((mid + 1) * cpf), :, :], outputs, normalized=True,
                                            raw=False)
            running_valid_psnr += valid_psnr
            valid_meters["valid_loss"].update(loss.item())
            valid_meters["valid_psnr"].update(valid_psnr.item())

            ######
            output_dir_img = os.path.join(output_dir_epoch, f"output_{sample_id}.png")
            transform_img = transforms.ToPILImage()
            img_save = np.array(transform_img(outputs[0].cpu().detach()))
            plt.imsave(output_dir_img, img_save)
            ######
            ######
            output_dir_img = os.path.join(output_dir_epoch, f"noisy_{sample_id}.png")
            transform_img = transforms.ToPILImage()
            img_save = np.array(transform_img(noisy_frame[0].cpu().detach()))
            plt.imsave(output_dir_img, img_save)
            ######

            running_valid_psnr /= (sample_id + 1)


def get_args():
    parser = argparse.ArgumentParser(allow_abbrev=False)

    # Add data arguments
    parser.add_argument("--data-path", default="datasets/noisy_0_crop", help="path to data directory")
    parser.add_argument("--noisy-path", default="datasets/noisy_1000_crop", help="path to data directory")
    parser.add_argument("--aug", default=0, type=int, help="augmentations")
    parser.add_argument("--batch-size", default=2, type=int, help="train batch size")
    parser.add_argument("--image-size", default=128, type=int, help="image size for train")
    parser.add_argument("--n-frames", default=5, type=int, help="number of frames for training")
    parser.add_argument("--stride", default=64, type=int, help="stride for patch extraction")

    # Add optimization arguments
    parser.add_argument("--lr", default=1e-4, type=float, help="learning rate")
    parser.add_argument("--num-epochs", default=1024, type=int, help="force stop training at specified epoch")
    parser.add_argument("--valid-interval", default=10, type=int, help="evaluate every N epochs")
    parser.add_argument("--save-interval", default=1, type=int, help="save a checkpoint every N steps")

    # add_logging_arguments
    parser.add_argument("--seed", default=0, type=int, help="random number generator seed")
    parser.add_argument("--output-dir", default="experiments", help="path to experiment directories")
    parser.add_argument("--experiment", default="RDVD", help="experiment name to be used with Tensorboard")
    parser.add_argument("--resume-training", action="store_false", help="whether to resume training")
    parser.add_argument("--no-save", action="store_true", help="don't save models or checkpoints")
    parser.add_argument("--step-checkpoints", action="store_true", help="store all step checkpoints")
    parser.add_argument("--no-log", action="store_true", help="don't save logs to file or Tensorboard directory")
    parser.add_argument("--log-interval", type=int, default=100, help="log every N steps")
    parser.add_argument("--no-visual", action="store_true", help="don't use Tensorboard")
    parser.add_argument("--visual-interval", type=int, default=100, help="log every N steps")
    parser.add_argument("--no-progress", action="store_true", help="don't use progress bar")
    parser.add_argument("--draft", action="store_true", help="save experiment results to draft directory")
    parser.add_argument("--dry-run", action="store_true", help="no log, no save, no visualization")

    parser.add_argument("--restore-file", default='experiments/RDVD-Apr-29-13_52_29/checkpoints', help="filename to load checkpoint")

    # Parse twice as model arguments are not known the first time
    BlindVideoNet.add_args(parser)
    args, _ = parser.parse_known_args()
    return args


if __name__ == "__main__":
    args = get_args()
    main(args)
