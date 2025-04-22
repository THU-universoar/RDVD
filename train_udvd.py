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
    utils.setup_experiment(args)
    utils.init_logging(args)

    # Build data loaders, a model and an optimizer
    model = BlindVideoNet(channels_per_frame=args.channels, out_channels=args.out_channels, bias=args.bias, blind=(not args.normal), sigma_known=(not args.blind_noise)).to(device)
    cpf = model.c # channels per frame
    mid = args.n_frames // 2
    model = nn.DataParallel(model)
    # print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1000], gamma=0.5)
    logging.info(f"Built a model consisting of {sum(p.numel() for p in model.parameters()):,} parameters")

    if args.resume_training:
        state_dict = utils.load_checkpoint(args, model, optimizer, scheduler)
        global_step = state_dict['last_step']
        start_epoch = int(state_dict['last_step']/(403200/state_dict['args'].batch_size))+1
    else:
        global_step = -1
        start_epoch = 0

    train_loader, valid_loader = load_SingleVideo(args.data_path, noisy_path=args.noisy_path, batch_size=args.batch_size, image_size=args.image_size, stride=args.stride, n_frames=args.n_frames, aug=args.aug)

    # Track moving average of loss values
    train_meters = {name: utils.RunningAverageMeter(0.98) for name in (["train_loss", "train_psnr", "train_ssim"])}
    if args.loss == "loglike":
        mean_meters = {name: utils.AverageMeter() for name in (["mean_psnr", "mean_ssim"])}
    valid_meters = {name: utils.AverageMeter() for name in (["valid_loss", "valid_psnr", "valid_ssim"])}
    writer = SummaryWriter(log_dir=args.experiment_dir) if not args.no_visual else None

########训练
    for epoch in range(start_epoch, args.num_epochs):
        if args.resume_training:
            if epoch %10 == 0:
                optimizer.param_groups[0]["lr"] /= 2
                print('learning rate reduced by factor of 2')

        train_bar = utils.ProgressBar(train_loader, epoch)
        for meter in train_meters.values():
            meter.reset()
        if args.loss == "loglike":
            for meter in mean_meters.values():
                meter.reset()

        for batch_id, (inputs, noisy_inputs) in enumerate(train_bar):

            # img_show = noisy_inputs[0, 6:9, :, :].permute(1, 2, 0).cpu().numpy()
            # plt.imshow(img_show)
            # plt.axis('off')
            # plt.show()
            # img_show = inputs[0, 6:9, :, :].permute(1, 2, 0).cpu().numpy()
            # plt.imshow(img_show)
            # plt.axis('off')
            # plt.show()

            model.train()
            global_step += 1
            inputs = inputs.to(device)
            noisy_inputs = noisy_inputs.to(device)
            
            outputs, est_sigma = model(noisy_inputs)
            noisy_frame = noisy_inputs[:, (mid*cpf):((mid+1)*cpf), :, :]
            truth_frame = inputs[:, (mid*cpf):((mid+1)*cpf), :, :]

            if args.blind_noise:
                loss = utils.loss_function(outputs, truth_frame, mode=args.loss, sigma=est_sigma, device=device)
            else:
                loss = utils.loss_function(outputs, truth_frame, mode=args.loss, sigma=args.noise_std/255, device=device)

            model.zero_grad()
            loss.backward()
            optimizer.step()

            if args.loss == "loglike":
                with torch.no_grad():
                    if args.blind_noise:
                        outputs, mean_image = utils.post_process(outputs, noisy_frame, sigma=est_sigma, device=device)
                    else:
                        outputs, mean_image = utils.post_process(outputs, noisy_frame, sigma=args.noise_std/255, device=device)

            train_psnr = utils.psnr(inputs[:, (mid*cpf):((mid+1)*cpf), :, :], outputs, normalized=True, raw=False)
            train_meters["train_loss"].update(loss.item())
            train_meters["train_psnr"].update(train_psnr.item())

            if args.loss == "loglike":
                mean_psnr = utils.psnr(inputs[:, (mid*cpf):((mid+1)*cpf), :, :], mean_image, normalized=True, raw=False)
                mean_meters["mean_psnr"].update(mean_psnr.item())

            if writer is not None and global_step % args.log_interval == 0:
                writer.add_scalar("lr", optimizer.param_groups[0]["lr"], global_step)
                writer.add_scalar("loss/train", loss.item(), global_step)
                writer.add_scalar("psnr/train", train_psnr.item(), global_step)
                if args.loss == "loglike":
                    writer.add_scalar("psnr/mean", mean_psnr.item(), global_step)
                gradients = torch.cat([p.grad.view(-1) for p in model.parameters() if p.grad is not None], dim=0)
                writer.add_histogram("gradients", gradients, global_step)
                sys.stdout.flush()

            if (batch_id+1) % 200 == 0:

                figure_dir = os.path.join(args.experiment_dir, "epoch", f"{epoch + 1}", f"batch{batch_id + 1}")
                os.makedirs(figure_dir, exist_ok=True)
                figure_dir_inputs = os.path.join(figure_dir, "inputs.png")
                figure_dir_outputs = os.path.join(figure_dir, "outputs.png")
                figure_dir_clean = os.path.join(figure_dir, "clean.png")
                transform_img = transforms.ToPILImage()
                inputs_save = np.array(transform_img(noisy_frame[0].cpu().detach()))
                outputs_save = np.array(transform_img(outputs[0].cpu().detach()))
                clean_save = np.array(transform_img(truth_frame[0].cpu().detach()))
                plt.imsave(figure_dir_inputs, inputs_save)
                plt.imsave(figure_dir_outputs, outputs_save)
                plt.imsave(figure_dir_clean, clean_save)

                if args.loss == "loglike":
                    logging.info(train_bar.print(dict(**train_meters, **mean_meters, lr=optimizer.param_groups[0]["lr"]))+f" | {batch_id+1} mini-batches ended")
                else:
                    logging.info(train_bar.print(dict(**train_meters, lr=optimizer.param_groups[0]["lr"]))+f" | {batch_id+1} mini-batches ended")
            if (batch_id+1) % 2000 == 0:
                model.eval()
                for meter in valid_meters.values():
                    meter.reset()
                if args.loss == "loglike":
                    for meter in mean_meters.values():
                        meter.reset()

                valid_bar = utils.ProgressBar(valid_loader)
                running_valid_psnr = 0.0
                for sample_id, (sample, noisy_inputs) in enumerate(valid_bar):
                    if args.heldout and (not sample_id == len(valid_loader.dataset)-3):
                        continue
                    with torch.no_grad():
                        sample = sample.to(device)
                        noisy_inputs = noisy_inputs.to(device)
                        outputs, est_sigma = model(noisy_inputs)
                        noisy_frame = noisy_inputs[:, (mid*cpf):((mid+1)*cpf), :, :]
                        truth_frame = sample[:, (mid*cpf):((mid+1)*cpf), :, :]

                        if args.blind_noise:
                            loss = utils.loss_function(outputs, truth_frame, mode=args.loss, sigma=est_sigma, device=device)
                        else:
                            loss = utils.loss_function(outputs, truth_frame, mode=args.loss, sigma=args.noise_std/255, device=device)

                        if args.loss == "loglike":
                            if args.blind_noise:
                                outputs, mean_image = utils.post_process(outputs, noisy_frame, sigma=est_sigma, device=device)
                            else:
                                outputs, mean_image = utils.post_process(outputs, noisy_frame, sigma=args.noise_std/255, device=device)

                        valid_psnr = utils.psnr(sample[:, (mid*cpf):((mid+1)*cpf), :, :], outputs, normalized=True, raw=False)
                        running_valid_psnr += valid_psnr
                        valid_meters["valid_loss"].update(loss.item())
                        valid_meters["valid_psnr"].update(valid_psnr.item())

                        if args.loss == "loglike":
                            mean_psnr = utils.psnr(sample[:, (mid*cpf):((mid+1)*cpf), :, :], mean_image, normalized=True, raw=False)
                            mean_meters["mean_psnr"].update(mean_psnr.item())

                running_valid_psnr /= (sample_id+1)

                if writer is not None:
                    writer.add_scalar("psnr/valid", valid_meters['valid_psnr'].avg, global_step)
                    sys.stdout.flush()

                if args.loss == "loglike":
                    logging.info("EVAL:"+train_bar.print(dict(**valid_meters, **mean_meters, lr=optimizer.param_groups[0]["lr"])))
                else:
                    logging.info("EVAL:"+train_bar.print(dict(**valid_meters, lr=optimizer.param_groups[0]["lr"])))
                utils.save_checkpoint(args, global_step, model, optimizer, score=valid_meters["valid_loss"].avg, mode="min")
        scheduler.step()

        if args.loss == "loglike":
            logging.info(train_bar.print(dict(**train_meters, **mean_meters, lr=optimizer.param_groups[0]["lr"])))
        else:
            logging.info(train_bar.print(dict(**train_meters, lr=optimizer.param_groups[0]["lr"])))
########验证
        if (epoch+1) % args.valid_interval == 0:

            #######设置存储路径
            output_dir_epoch = os.path.join(args.experiment_dir, "valid", f"epoch_{epoch+1}")
            os.makedirs(output_dir_epoch, exist_ok=True)
            #######

            model.eval()
            for meter in valid_meters.values():
                meter.reset()
            if args.loss == "loglike":
                for meter in mean_meters.values():
                    meter.reset()

            valid_bar = utils.ProgressBar(valid_loader)
            running_valid_psnr = 0.0
            for sample_id, (sample, noisy_inputs) in enumerate(valid_bar):

                with (torch.no_grad()):
                    sample = sample.to(device)
                    noisy_inputs = noisy_inputs.to(device)
                    outputs, est_sigma = model(noisy_inputs)
                    noisy_frame = noisy_inputs[:, (mid*cpf):((mid+1)*cpf), :, :]
                    truth_frame = sample[:, (mid*cpf):((mid+1)*cpf), :, :]

                    if args.blind_noise:
                        loss = utils.loss_function(outputs, truth_frame, mode=args.loss, sigma=est_sigma, device=device)
                    else:
                        loss = utils.loss_function(outputs, truth_frame, mode=args.loss, sigma=args.noise_std/255, device=device)

                    if args.loss == "loglike":
                        if args.blind_noise:
                            outputs, mean_image = utils.post_process(outputs, noisy_frame, sigma=est_sigma, device=device)
                        else:
                            outputs, mean_image = utils.post_process(outputs, noisy_frame, sigma=args.noise_std/255, device=device)

                    valid_psnr = utils.psnr(sample[:, (mid*cpf):((mid+1)*cpf), :, :], outputs, normalized=True, raw=False)
                    running_valid_psnr += valid_psnr
                    valid_meters["valid_loss"].update(loss.item())
                    valid_meters["valid_psnr"].update(valid_psnr.item())

                    if args.loss == "loglike":
                        mean_psnr = utils.psnr(sample[:, (mid*cpf):((mid+1)*cpf), :, :], mean_image, normalized=True, raw=False)
                        mean_meters["mean_psnr"].update(mean_psnr.item())

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
                    ######
                    output_dir_img = os.path.join(output_dir_epoch, f"clean_{sample_id}.png")
                    transform_img = transforms.ToPILImage()
                    img_save = np.array(transform_img(truth_frame[0].cpu().detach()))
                    plt.imsave(output_dir_img, img_save)
                    ######


            running_valid_psnr /= (sample_id+1)

            if writer is not None:
                writer.add_scalar("psnr/valid", valid_meters['valid_psnr'].avg, global_step)
                sys.stdout.flush()

            if args.loss == "loglike":
                logging.info("EVAL:"+train_bar.print(dict(**valid_meters, **mean_meters, lr=optimizer.param_groups[0]["lr"])))
            else:
                logging.info("EVAL:"+train_bar.print(dict(**valid_meters, lr=optimizer.param_groups[0]["lr"])))
            utils.save_checkpoint(args, global_step, model, optimizer, score=valid_meters["valid_loss"].avg, mode="min")


    logging.info(f"Done training! Best PSNR {utils.save_checkpoint.best_score:.3f} obtained after step {utils.save_checkpoint.best_step}.")


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
    # Add loss function
    parser.add_argument("--loss", default="loglike", help="loss function used for training")

    # Add optimization arguments
    parser.add_argument("--lr", default=1e-4, type=float, help="learning rate")
    parser.add_argument("--num-epochs", default=1024, type=int, help="force stop training at specified epoch")
    parser.add_argument("--valid-interval", default=10, type=int, help="evaluate every N epochs")
    parser.add_argument("--save-interval", default=1, type=int, help="save a checkpoint every N steps")

    #add_logging_arguments
    parser.add_argument("--seed", default=0, type=int, help="random number generator seed")
    parser.add_argument("--output-dir", default="experiments", help="path to experiment directories")
    parser.add_argument("--experiment", default="RDVD", help="experiment name to be used with Tensorboard")
    parser.add_argument("--resume-training", action="store_true", help="whether to resume training")
    parser.add_argument("--restore-file", default=None, help="filename to load checkpoint")
    parser.add_argument("--no-save", action="store_true", help="don't save models or checkpoints")
    parser.add_argument("--step-checkpoints", action="store_true", help="store all step checkpoints")
    parser.add_argument("--no-log", action="store_true", help="don't save logs to file or Tensorboard directory")
    parser.add_argument("--log-interval", type=int, default=100, help="log every N steps")
    parser.add_argument("--no-visual", action="store_true", help="don't use Tensorboard")
    parser.add_argument("--visual-interval", type=int, default=100, help="log every N steps")
    parser.add_argument("--no-progress", action="store_true", help="don't use progress bar")
    parser.add_argument("--draft", action="store_true", help="save experiment results to draft directory")
    parser.add_argument("--dry-run", action="store_true", help="no log, no save, no visualization")

    # Parse twice as model arguments are not known the first time
    BlindVideoNet.add_args(parser)
    args, _ = parser.parse_known_args()
    return args


if __name__ == "__main__":
    args = get_args()
    main(args)
