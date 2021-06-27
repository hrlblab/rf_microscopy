from typing import List, Union
import scipy.fftpack as ft
import scipy.signal as sig
from scipy.stats import norm
import matplotlib.pyplot as plt
import math
from PIL import Image
import pandas as pd
from scipy.signal import convolve2d
import torch
from torch import nn
from torch.nn import functional as F
from torch_metrics import compute_ssim, compute_psnr
from utils import (add_mask_params, save_json, build_optim, count_parameters,
                               count_trainable_parameters, count_untrainable_parameters, str2bool, str2none)
from policy_model_def import build_policy_model
from data_loading import create_data_loader
import copy
import random
import logging
import time
import copy
import datetime
import random
import argparse
import pathlib
import wandb
from random import choice
from string import ascii_uppercase
import torch
import numpy as np
from tensorboardX import SummaryWriter
import tensorflow as tf


def compute_backprop_trajectory(args, kspace, masked_kspace, mask, unnorm_gt, recons, gt_mean, gt_std,
                                data_range, model, recon_model, step, action_list, logprob_list, reward_list):
    # Base score from which to calculate acquisition rewards
    base_score = compute_scores(recon_model)
    # Get policy and probabilities.
    policy, probs = get_policy_probs(model, recons, mask)
    # Sample actions from the policy. For greedy (or at step = 0) we sample num_trajectories actions from the
    # current policy. For non-greedy with step > 0, we sample a single action for every of the num_trajectories
    # policies.
    # probs shape = batch x num_traj x res
    # actions shape = batch x num_traj
    # action_logprobs shape = batch x num_traj
    if step == 0 or args.model_type == 'greedy':  # probs has shape batch x 1 x res
        actions = torch.multinomial(probs.squeeze(1), args.num_trajectories, replacement=True)
        actions = actions.unsqueeze(1)  # batch x num_traj -> batch x 1 x num_traj
        # probs shape = batch x 1 x res
        action_logprobs = torch.log(torch.gather(probs, -1, actions)).squeeze(1)
        actions = actions.squeeze(1)
    else:  # Non-greedy model and step > 0: this means probs has shape batch x num_traj x res
        actions = policy.sample()
        actions = actions.unsqueeze(-1)  # batch x num_traj -> batch x num_traj x 1
        # probs shape = batch x num_traj x res
        action_logprobs = torch.log(torch.gather(probs, -1, actions)).squeeze(-1)
        actions = actions.squeeze(1)

    # Obtain rewards in parallel by taking actions in parallel
    mask, masked_kspace, recons = compute_next_step_reconstruction(recon_model, kspace,
                                                                       masked_kspace, mask, actions)
    ssim_scores = compute_scores(recon_model)
    # batch x num_trajectories
    action_rewards = ssim_scores - base_score
    # batch x 1
    avg_reward = torch.mean(action_rewards, dim=-1, keepdim=True)
    # Store for non-greedy model (we need the full return before we can do a backprop step)
    action_list.append(actions)
    logprob_list.append(action_logprobs)
    reward_list.append(action_rewards)

    if args.model_type == 'greedy':
        # batch x k
        if args.no_baseline:
            # No-baseline
            loss = -1 * (action_logprobs * action_rewards) / actions.size(-1)
        else:
            # Local baseline
            loss = -1 * (action_logprobs * (action_rewards - avg_reward)) / (actions.size(-1) - 1)
        # batch
        loss = loss.sum(dim=1)
        # Average over batch
        # Divide by batches_step to mimic taking mean over larger batch
        loss = loss.mean() / args.batches_step  # For consistency: we generally set batches_step to 1 for greedy
        loss.backward()

        # For greedy: initialise next step by randomly picking one of the measurements for every slice
        # For non-greedy we will continue with the parallel sampled rows stored in masked_kspace, and
        # with mask, zf, and recons.
        idx = random.randint(0, mask.shape[1] - 1)
        mask = mask[:, idx:idx + 1, :, :, :]
        masked_kspace = masked_kspace[:, idx:idx + 1, :, :, :]
        recons = recons[:, idx:idx + 1, :, :]

    elif step != args.acquisition_steps - 1:  # Non-greedy but don't have full return yet.
        loss = torch.zeros(1)  # For logging
    else:  # Final step, can compute non-greedy return
        reward_tensor = torch.stack(reward_list)
        for step, logprobs in enumerate(logprob_list):
            # Discount factor
            gamma_vec = [args.gamma ** (t - step) for t in range(step, args.acquisition_steps)]
            gamma_ten = torch.tensor(gamma_vec).unsqueeze(-1).unsqueeze(-1).to(args.device)
            # step x batch x 1
            avg_rewards_tensor = torch.mean(reward_tensor, dim=2, keepdim=True)
            # Get number of trajectories for correct average
            num_traj = logprobs.size(-1)
            # REINFORCE with self-baselines
            # batch x k
            # TODO: can also store transitions (s, a, r, s') pairs and recompute log probs when
            #  doing gradients? Takes less memory, but more compute: can this be efficiently
            #  batched?
            loss = -1 * (logprobs * torch.sum(
                gamma_ten * (reward_tensor[step:, :, :] - avg_rewards_tensor[step:, :, :]),
                dim=0)) / (num_traj - 1)
            # batch
            loss = loss.sum(dim=1)
            # Average over batch
            # Divide by batches_step to mimic taking mean over larger batch
            loss = loss.mean() / args.batches_step
            loss.backward()  # Store gradients

    return loss, mask, masked_kspace, recons


def train_epoch(args, epoch, recon_model, model, loader, optimiser, writer, data_range_dict):
    """
    Performs a single training epoch.

    :param args: Argument object, containing hyperparameters for model training.
    :param epoch: int, current training epoch.
    :param recon_model: reconstruction model object.
    :param model: policy model object.
    :param loader: training data loader.
    :param optimiser: PyTorch optimizer.
    :param writer: Tensorboard writer.
    :param data_range_dict: dictionary containing the dynamic range of every volume in the training data.
    :return: (float: mean loss of this epoch, float: epoch duration)
    """
    model.train()
    epoch_loss = [0. for _ in range(args.acquisition_steps)]
    report_loss = [0. for _ in range(args.acquisition_steps)]
    start_epoch = start_iter = time.perf_counter()
    global_step = epoch * len(loader)

    cbatch = 0  # Counter for spreading single backprop batch over multiple data loader batches
    for it, data in enumerate(loader):  # Loop over data points
        cbatch += 1
        kspace, masked_kspace, mask, zf, gt, gt_mean, gt_std, fname, _ = data
        # shape after unsqueeze = batch x channel x columns x rows x complex
        kspace = kspace.unsqueeze(1).to(args.device)
        masked_kspace = masked_kspace.unsqueeze(1).to(args.device)
        mask = mask.unsqueeze(1).to(args.device)
        # shape after unsqueeze = batch x channel x columns x rows
        zf = zf.unsqueeze(1).to(args.device)
        gt = gt.unsqueeze(1).to(args.device)
        gt_mean = gt_mean.unsqueeze(1).unsqueeze(2).unsqueeze(3).to(args.device)
        gt_std = gt_std.unsqueeze(1).unsqueeze(2).unsqueeze(3).to(args.device)
        unnorm_gt = gt * gt_std + gt_mean  # Unnormalise ground truth image for SSIM calculations
        data_range = torch.stack([data_range_dict[vol] for vol in fname])  # For SSIM calculations

        # Base reconstruction model forward pass: input to policy model
        recons = recon_model(zf)

        if cbatch == 1:  # Only after backprop is performed
            optimiser.zero_grad()

        action_list = []
        logprob_list = []
        reward_list = []
        for step in range(args.acquisition_steps):  # Loop over acquisition steps
            loss, mask, masked_kspace, recons = compute_backprop_trajectory(args, kspace, masked_kspace, mask,
                                                                            unnorm_gt, recons, gt_mean, gt_std,
                                                                            data_range, model, recon_model, step,
                                                                            action_list, logprob_list, reward_list)
            # Loss logging
            epoch_loss[step] += loss.item() / len(loader) * gt.size(0) / args.batch_size
            report_loss[step] += loss.item() / args.report_interval * gt.size(0) / args.batch_size
            writer.add_scalar('TrainLoss_step{}'.format(step), loss.item(), global_step + it)

        # Backprop if we've reached the prerequisite number of dataloader batches
        if cbatch == args.batches_step:
            optimiser.step()
            cbatch = 0

        # Logging: note that loss values mean little, as the Policy Gradient loss is not a true loss.
        if it % args.report_interval == 0:
            if it == 0:
                loss_str = ", ".join(["{}: {:.2f}".format(i + 1, args.report_interval * l * 1e3)
                                      for i, l in enumerate(report_loss)])
            else:
                loss_str = ", ".join(["{}: {:.2f}".format(i + 1, l * 1e3) for i, l in enumerate(report_loss)])
            logging.info(
                f'Epoch = [{epoch:3d}/{args.num_epochs:3d}], '
                f'Iter = [{it:4d}/{len(loader):4d}], '
                f'Time = {time.perf_counter() - start_iter:.2f}s, '
                f'Avg Loss per step x1e3 = [{loss_str}] ',
            )
            report_loss = [0. for _ in range(args.acquisition_steps)]

        start_iter = time.perf_counter()

    if args.wandb:
        wandb.log({'train_loss_step': {str(key + 1): val for key, val in enumerate(epoch_loss)}}, step=epoch + 1)

    return np.mean(epoch_loss), time.perf_counter() - start_epoch


def createAnnulus(n=256, r=32, w=4):
    ''' createAnnulus - create a ring-like structure
    INPUT
    n - size of square array or vector
    r - radius of the ring
    w - width of the ring
    OUTPUT
    an array n x n
    '''
    if np.isscalar(n):
        v = np.arange(n)
        v = v - np.floor(n / 2)
    else:
        v = n

    y, x = np.meshgrid(v, v)
    q = np.hypot(x, y)
    annulus = abs(q - r) < w

    return annulus


def doConventionalScan(Fsqmod, Lsqmod):
    '''Simulate Conventional digital scanning / dithering
        INPUT
        F_sqmod - Square modulus of F at the front focal plane
        L_sqmod - Square modulus of L at the front focal plane
        OUTPUT
        scanned - Scanned (dithered) intensity of Fsqmod by Lsqmod
    '''
    # Manually scan by shifting Fsqmod and multiplying by Lsqmod
    scanned = np.zeros(Fsqmod.shape)
    center = Lsqmod.shape[1] // 2

    for x in range(np.size(Fsqmod, 1)):
        scanned = scanned + np.roll(Fsqmod, x - center, 1) * Lsqmod[center, x]

    return scanned


def doConventionalScanHat(F_hat, L_hat):
    '''Simulate Conventional digital scanning / dithering from frequency space representations
       INPUT
       F_hat - Mask at back focal plane
       L_hat - Line scan profile in frequency space at the back focal plane
       OUTPUT
       scanned - Scanned (dithered) intensity of Fsqmod by Lsqmod at front focal plane
    '''
    F_hat = ft.ifftshift(F_hat)
    F = ft.ifft2(F_hat)
    F = ft.fftshift(F)
    # This is the illumination intensity pattern
    Fsqmod = np.real(F * np.conj(F))

    L_hat = ft.ifftshift(L_hat)
    L = ft.ifft2(L_hat)
    L = ft.fftshift(L)
    Lsqmod = L * np.conj(L)

    scanned = doConventionalScan(Fsqmod, Lsqmod)
    return scanned


def doFieldSynthesisLineScan(F_hat, L_hat):
    '''Simulate Field Synthesis Method
        INPUT
        F_hat - Frequency space representation of illumination pattern, mask at back focal plane
        L_hat - Line scan profile in frequency space at the back focal plane
        OUTPUT
        fieldSynthesis - Field synthesis construction by doing a line scan in the back focal plane
    '''
    # Do the Field Synthesis method of performing a line scan at the back focal plane
    fieldSynthesis = np.zeros_like(F_hat)

    for a in range(fieldSynthesis.shape[1]):
        # Instaneous scan in frequency space
        T_hat_a = F_hat * np.roll(L_hat, a - fieldSynthesis.shape[1] // 2, 1)
        # Instaneous scan in object space
        T_a = ft.fftshift(ft.fft2(ft.ifftshift(T_hat_a)))
        # Incoherent summing of the intensities
        fieldSynthesis = fieldSynthesis + np.abs(T_a) ** 2

    return fieldSynthesis


def conv2(x, y, mode='same'):
    return np.rot90(convolve2d(np.rot90(x, 2), np.rot90(y, 2), mode=mode), 2)


def osWidth_gui(prof):
    # df = pd.DataFrame(prof)

    # gives no. of rows along x-axis
    prof = prof.reshape((len(prof), 1))

    e = np.exp(1)

    # sum over rows for each of the column
    prof2 = np.cumsum(prof, axis=0)
    # np.amax(prof) = Maximum of the flattened array
    prof2_n = prof2 / np.amax(prof2)

    # Find the indices of the maximum values along each column
    p_maxint = np.argmax(prof, axis=0)

    num_par_y_array_size = 4096
    ny = num_par_y_array_size

    totalarea = prof.sum(axis=0)
    guessthickness_pxl = 3
    thickness63percent = 0

    while thickness63percent == 0:
        guessthickness_pxl = guessthickness_pxl + 1
        if guessthickness_pxl == ny:
            thickness63percent = float('nan')
        # csum_norm is a 1xguessthickness_pxl array with all zeros
        # csum_norm = np.zeros(1, guessthickness_pxl)
        csum_norm = np.zeros((1, ny))
        guessstartpoint = max(p_maxint - guessthickness_pxl + 1, 1)
        guessendpoint = min(p_maxint + guessthickness_pxl - 1, ny) - guessthickness_pxl

        for ii in range(guessstartpoint[0], guessendpoint[0] + 1):
            prof_partial = np.array(prof)
            indices = list(range(ii, ii + guessthickness_pxl + 1))
            prof_partial_sum = prof_partial[indices].sum()
            csum_norm[0, ii] = prof_partial_sum / totalarea
            if not thickness63percent and csum_norm[0, ii] >= (1 - 1 / e):
                thickness63percent = guessthickness_pxl

    return thickness63percent * 1


def osWidth_gui_2(prof):
    # df = pd.DataFrame(prof)

    e = np.exp(1)

    # sum over rows for each of the column
    prof_row = np.sum(prof, axis=1)

    num_par_y_array_size = 4096
    ny = num_par_y_array_size

    totalarea = prof_row.sum(axis=0)
    guessthickness_pxl = 0
    thickness63percent = 0

    while thickness63percent == 0:
        # csum_norm is a 1xguessthickness_pxl array with all zeros
        # csum_norm = np.zeros(1, guessthickness_pxl)
        csum_norm = np.zeros((1, ny))

        for ii in range(1, 2048):
            guessthickness_pxl = guessthickness_pxl + 2
            prof_partial = np.array(prof_row)
            indices = list(range(2048 - ii, 2048 + ii + 1))
            prof_partial_sum = prof_partial[indices].sum()
            csum_norm[0, ii] = prof_partial_sum / totalarea
            if not thickness63percent and csum_norm[0, ii] >= (1 - 1 / e):
                thickness63percent = guessthickness_pxl

    return thickness63percent


def analysisbeam_gui(PSFSumY):
    cy = 2048
    centerline = PSFSumY[cy - 1]
    num_par_dx = 1
    pMaxPos = pWidth_gui(centerline, num_par_dx)

    prof_peak = PSFSumY[:][pMaxPos - 1]
    os_peak_e = osWidth_gui_2(prof_peak)

    return os_peak_e


def pWidth_gui(prof, dy):
    df = pd.DataFrame(prof)

    # gives no. of rows along x-axis
    if len(df) == 1:
        prof = np.transpose(prof)

    # Find the indices of the maximum values along each column
    pMaxPos = np.argmax(prof, axis=0)

    return pMaxPos


def createAnnulus(n=256, r=32, w=4):
    """
    createAnnulus - create a ring-like structure
    INPUT
    n - size of square array or vector
    r - radius of the ring
    w - width of the ring
    OUTPUT
    an array n x n
    """
    if np.isscalar(n):
        v = np.arange(n)
        v = v - np.floor(n / 2)
    else:
        v = n

    y, x = np.meshgrid(v, v)
    q = np.hypot(x, y)
    annulus = abs(q - r) < w

    return annulus


def compute_kspace(width):
    n = 4096
    r = 256

    dispRange: List[Union[int, float]] = []
    for i in range(-600, 601):
        dispRange.append(i + math.floor(n / 2) + 1)
    v = []
    for i in range(0, n):
        v.append(i - math.floor(n / 2))
    kspace = createAnnulus(v, r, width)

    return kspace


def compute_mask(w, pos):
    offset = 256
    n = 4096
    v = []
    for i in range(0, n):
        v.append(i - math.floor(n / 2))
    initial_v = []
    for i in range(0, n):
        initial_v.append(-v[i])
    mask = []
    for i in range(0, n):
        if (offset * 0.99 / 1.35 + pos + w / 2 > initial_v[i] > offset * 0.99 / 1.35 + pos - w / 2) or \
                (offset * 0.99 / 1.35 - pos + w / 2 > initial_v[i] > offset * 0.99 / 1.35 - pos - w / 2):
            mask.append(1)
        else:
            mask.append(0)
    return mask


def compute_masked_kspace(kspace, mask):
    n = 4096
    masked_kspace = kspace

    for c in range(0, n):
        if not mask[c]:
            masked_kspace[:, c] = False
    masked_kspace = masked_kspace.astype(float)
    return masked_kspace


def compute_scores(recon):
    # df = pd.DataFrame(prof)
    e = np.exp(1)

    # sum over rows for each of the column
    prof_row = np.sum(recon, axis=1)

    num_par_y_array_size = 4096
    ny = num_par_y_array_size

    totalarea = prof_row.sum(axis=0)
    guessthickness_pxl = 0
    thickness63percent = 0

    while thickness63percent == 0:
        # csum_norm is a 1xguessthickness_pxl array with all zeros
        # csum_norm = np.zeros(1, guessthickness_pxl)
        csum_norm = np.zeros((1, ny))

        for ii in range(1, 2048):
            guessthickness_pxl = guessthickness_pxl + 2
            prof_partial = np.array(prof_row)
            indices = list(range(2048 - ii, 2048 + ii + 1))
            prof_partial_sum = prof_partial[indices].sum()
            csum_norm[0, ii] = prof_partial_sum / totalarea
            if not thickness63percent and csum_norm[0, ii] >= (1 - 1 / e):
                thickness63percent = guessthickness_pxl

    return thickness63percent


def load_recon_model(width, position):
    n = 4096
    offset = 256
    v = []
    for i in range(0, n):
        v.append(i - math.floor(n / 2))

    kspace = compute_kspace(width)
    mask = compute_mask(width, position)
    masked_kspace = compute_masked_kspace(kspace, mask)

    fieldSynthesisProfile = ft.fftshift(ft.ifft(ft.ifftshift(masked_kspace)))
    df = pd.DataFrame(fieldSynthesisProfile)
    lattice_efield = ft.fftshift(ft.ifft2(ft.ifftshift(masked_kspace)))
    lattice = abs(lattice_efield) ** 2
    lattice_hat = ft.fftshift(ft.fft2(ft.ifftshift(lattice)))

    period = n / offset
    if period == round(period):
        recon = conv2(lattice, np.ones((1, int(period))) / period, 'same')
    else:
        for x in lattice_hat:
            for y in x:
                recon = y * (np.sinc(v / period))
        recon = ft.fftshift(ft.ifft2(ft.ifftshift(recon)))
    return recon


def load_policy_model(checkpoint_file, optim=False):
    checkpoint = torch.load(checkpoint_file)
    args = checkpoint['args']
    model = build_policy_model(args)

    if not optim:
        # No gradients for this model
        for param in model.parameters():
            param.requires_grad = False

    if args.data_parallel:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(checkpoint['model'])

    start_epoch = checkpoint['epoch']

    if optim:
        optimizer = build_optim(args, model.parameters())
        optimizer.load_state_dict(checkpoint['optimizer'])
        del checkpoint
        return model, args, start_epoch, optimizer

    del checkpoint
    return model, args


# 当前width 49
# 给一个probs -> 50 -> 从-25到25
def get_policy_probs(model, recons, mask):
    channel_size = mask.shape[1]
    res = mask.size(-2)
    # Reshape trajectory dimension into batch dimension for parallel forward pass
    recons = recons.view(mask.size(0) * channel_size, 1, res, res)
    # Obtain policy model logits
    output = model(recons)
    # Reshape trajectories back into their own dimension
    output = output.view(mask.size(0), channel_size, res)
    # Mask already acquired rows by setting logits to very negative numbers
    loss_mask = (mask == 0).squeeze(-1).squeeze(-2).float()
    logits = torch.where(loss_mask.byte(), output, -1e7 * torch.ones_like(output))
    # Softmax over 'logits' representing row scores
    probs = torch.nn.functional.softmax(logits - logits.max(dim=-1, keepdim=True)[0], dim=-1)
    # Also need this for sampling the next row at the end of this loop
    policy = torch.distributions.Categorical(probs)
    return policy, probs


def create_arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--resolution', default=4096, type=int, help='Resolution of images')
    parser.add_argument('--filters', type=int, default=5, help='Number of convolution kernels。')
    parser.add_argument('--dataset', default='knee', help='Dataset type to use.')
    parser.add_argument('--data_path', type=pathlib.Path, required=True,
                        help="Path to the dataset. Make sure to set this consistently with the 'dataset' "
                             "argument above.")
    parser.add_argument('--sample_rate', type=float, default=0.5,
                        help='Fraction of total volumes to include')
    parser.add_argument('--acquisition', type=str2none, default=None,
                        help='Use only volumes acquired using the provided acquisition method. Options are: '
                             'CORPD_FBK, CORPDFS_FBK (fat-suppressed), and not provided (both used).')
    parser.add_argument('--report_interval', type=int, default=1000, help='Period of loss reporting')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers to use for data loading')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Which device to train on. Set to "cuda" to use the GPU')
    parser.add_argument('--exp_dir', type=pathlib.Path, default=None,
                        help='Directory where model and results should be saved. Will create a timestamped folder '
                             'in provided directory each run')
    parser.add_argument('--accelerations', nargs='+', default=[8], type=int,
                        help='Ratio of k-space columns to be sampled. If multiple values are '
                             'provided, then one of those is chosen uniformly at random for '
                             'each volume.')
    parser.add_argument('--reciprocals_in_center', nargs='+', default=[1], type=float,
                        help='Inverse fraction of rows (after subsampling) that should be in the center. E.g. if half '
                             'of the sampled rows should be in the center, this should be set to 2. All combinations '
                             'of acceleration and reciprocals-in-center will be used during training (every epoch a '
                             'volume randomly gets assigned an acceleration and center fraction.')
    parser.add_argument('--acquisition_steps', default=16, type=int, help='Acquisition steps to train for per image.')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of ConvNet layers. Note that setting '
                                                                  'this too high will cause size mismatch errors, due to even-odd errors in calculation for '
                                                                  'layer size post-flattening (due to max pooling).')
    parser.add_argument('--drop_prob', type=float, default=0, help='Dropout probability')
    parser.add_argument('--batch_size', default=16, type=int, help='Mini batch size for training')
    parser.add_argument('--val_batch_size', default=64, type=int, help='Mini batch size for validation')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='Strength of weight decay regularization.')
    parser.add_argument('--center_volume', type=str2bool, default=True,
                        help='If set, only the center slices of a volume will be included in the dataset. This '
                             'removes the most noisy images from the data.')
    parser.add_argument('--data_parallel', type=str2bool, default=True,
                        help='If set, use multiple GPUs using data parallelism')
    parser.add_argument('--do_train_ssim', type=str2bool, default=False,
                        help='Whether to compute SSIM values on training data.')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='Number of training epochs')  # wenying change epochs test
    parser.add_argument('--seed', default=0, type=int, help='Seed for random number generators. '
                                                            'Set to 0 to use random seed.')
    parser.add_argument('--num_chans', type=int, default=16, help='Number of ConvNet channels in first layer.')
    parser.add_argument('--fc_size', default=256, type=int, help='Size (width) of fully connected layer(s).')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--lr_gamma', type=float, default=0.1,
                        help='Multiplicative factor of learning rate decay')
    parser.add_argument('--scheduler_type', type=str, choices=['step', 'multistep'], default='step',
                        help='Number of training epochs')
    parser.add_argument('--lr_step_size', type=int, default=40,
                        help='Period of learning rate decay')
    parser.add_argument('--lr_multi_step_size', nargs='+', type=int, default=[10, 20, 30, 40],
                        help='Epoch at which to decay the lr if using multistep scheduler.')
    parser.add_argument('--model_type', type=str, default='greedy', choices=['greedy', 'nongreedy'],
                        help="'greedy' to train greedy model, 'nongreedy' to train non-greedy model")
    parser.add_argument('--batches_step', type=int, default=1,
                        help='Number of dataloader batches to compute before doing an optimiser step. This is mostly '
                             'used to train non-greedy models with larger batch sizes.')
    parser.add_argument('--no_baseline', type=str2bool, default=False,
                        help="Whether to not use a reward baseline at all. Currently only implemented for 'greedy'.")
    parser.add_argument('--gamma', type=float, default=1,
                        help='Discount factor in RL. Currently only used for non-greedy training.')
    parser.add_argument('--milestones', nargs='+', type=int, default=[0, 9, 19, 29, 39, 49],
                        help='Epochs at which to save model separately.')

    parser.add_argument('--do_train', type=str2bool, default=True,
                        help='Whether to do training or testing.')
    parser.add_argument('--policy_model_checkpoint', type=pathlib.Path, default=None,
                        help='Path to a pretrained policy model if do_train is False (testing).')

    parser.add_argument('--wandb', type=str2bool, default=False,
                        help='Whether to use wandb logging for this run.')
    parser.add_argument('--project', type=str2none, default=None,
                        help='Wandb project name to use.')

    parser.add_argument('--resume', type=str2bool, default=False,
                        help='Continue training previous run?')
    parser.add_argument('--run_id', type=str2none, default=None,
                        help='Wandb run_id to continue training from.')

    parser.add_argument('--num_test_trajectories', type=int, default=1,
                        help='Number of trajectories to use when testing sampling policy.')
    parser.add_argument('--test_multi', type=str2bool, default=False,
                        help='Test multiple models in one script')
    parser.add_argument('--policy_model_list', nargs='+', type=str, default=[None],
                        help='List of policy model paths for multi-testing.')

    return parser


def create_data_range_dict(args, loader):
    # Locate ground truths of a volume
    gt_vol_dict = {}
    for it, data in enumerate(loader):
        kspace, masked_kspace, mask, zf, gt, gt_mean, gt_std, fname, slice = data
        for i, vol in enumerate(fname):
            if vol not in gt_vol_dict:
                gt_vol_dict[vol] = []
            gt_vol_dict[vol].append(gt[i] * gt_std[i] + gt_mean[i])
    # Find max of a volume
    data_range_dict = {}
    for vol, gts in gt_vol_dict.items():
        # Shape 1 x 1 x 1 x 1
        data_range_dict[vol] = torch.stack(gts).max().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(args.device)
    del gt_vol_dict
    return data_range_dict


def train_and_eval(args, recon_args, recon_model):
    """
    Wrapper for training and evaluation of policy model.

    :param args: Argument object, containing hyperparameters for training and evaluation.
    :param recon_args: reconstruction model arguments.
    :param recon_model: reconstruction model.
    """
    """
    Wrapper for training and evaluation of policy model.

    :param args: Argument object, containing hyperparameters for training and evaluation.
    :param recon_args: reconstruction model arguments.
    :param recon_model: reconstruction model.
    """
    # Load previously built policy network
    if args.resume:
        # Check that this works
        resumed = True
        new_run_dir = args.policy_model_checkpoint.parent
        data_path = args.data_path
        # In case models have been moved to a different machine, make sure the path to the recon model is the
        # path provided.
        recon_model_checkpoint = args.recon_model_checkpoint

        model, args, start_epoch, optimiser = load_policy_model(pathlib.Path(args.policy_model_checkpoint), optim=True)

        args.old_run_dir = args.run_dir
        args.old_recon_model_checkpoint = args.recon_model_checkpoint
        args.old_data_path = args.data_path

        args.recon_model_checkpoint = recon_model_checkpoint
        args.run_dir = new_run_dir
        args.data_path = data_path
        args.resume = True
    else: #what we run!!!!
        resumed = False
        # Improvement model to train
        model = build_policy_model(args)
        # Add mask parameters for training
        # args = add_mask_params(args)
        # if args.data_parallel:
        #     model = torch.nn.DataParallel(model)
        optimiser = build_optim(args, model.parameters())
        start_epoch = 0
        # Create directory to store results in
        # savestr = '{}_res{}_al{}_accel{}_k{}_{}_{}'.format(args.dataset, args.resolution, args.acquisition_steps,
        #                                                    args.accelerations, args.num_trajectories,
        #                                                    datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S"),
        #                                                    ''.join(choice(ascii_uppercase) for _ in range(5)))

        savestr = '{}_res{}_al{}_accel{}_k{}_{}'.format(args.dataset, args.resolution, args.acquisition_steps,
                                                           args.accelerations, args.num_trajectories,
                                                           ''.join(choice(ascii_uppercase) for _ in range(5)))
        # # args.run_dir = args.exp_dir / savestr
        # args.run_dir.mkdir(parents=True, exist_ok=False)

    args.resumed = resumed

    if args.wandb:
        allow_val_change = args.resumed  # only allow changes if resumed: otherwise something is wrong.
        wandb.config.update(args, allow_val_change=allow_val_change)
        wandb.watch(model, log='all')

    # Logging
    logging.info(recon_model)
    logging.info(model)
    # Save arguments for bookkeeping
    args_dict = {key: str(value) for key, value in args.__dict__.items()
                 if not key.startswith('__') and not callable(key)}
    save_json(args.run_dir / 'args.json', args_dict)

    # Initialise summary writer
    writer = SummaryWriter(log_dir=args.run_dir / 'summary')

    # Parameter counting
    logging.info('Reconstruction model parameters: total {}, of which {} trainable and {} untrainable'.format(
        count_parameters(recon_model), count_trainable_parameters(recon_model),
        count_untrainable_parameters(recon_model)))
    logging.info('Policy model parameters: total {}, of which {} trainable and {} untrainable'.format(
        count_parameters(model), count_trainable_parameters(model), count_untrainable_parameters(model)))

    if args.scheduler_type == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimiser, args.lr_step_size, args.lr_gamma)
    elif args.scheduler_type == 'multistep':
        if not isinstance(args.lr_multi_step_size, list):
            args.lr_multi_step_size = [args.lr_multi_step_size]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimiser, args.lr_multi_step_size, args.lr_gamma)
    else:
        raise ValueError("{} is not a valid scheduler choice ('step', 'multistep')".format(args.scheduler_type))

    # Create data loaders
    train_loader = create_data_loader(args, 'train', shuffle=True)
    dev_loader = create_data_loader(args, 'val', shuffle=False)

    train_data_range_dict = create_data_range_dict(args, train_loader)
    dev_data_range_dict = create_data_range_dict(args, dev_loader)

    if not args.resume:
        if args.do_train_ssim:
            do_and_log_evaluation(args, -1, recon_model, model, train_loader, writer, 'Train', train_data_range_dict)
        do_and_log_evaluation(args, -1, recon_model, model, dev_loader, writer, 'Val', dev_data_range_dict)

    for epoch in range(start_epoch, args.num_epochs):
        train_loss, train_time = train_epoch(args, epoch, recon_model, model, train_loader, optimiser, writer,
                                             train_data_range_dict)
        logging.info(
            f'Epoch = [{epoch+1:3d}/{args.num_epochs:3d}] TrainLoss = {train_loss:.3g} TrainTime = {train_time:.2f}s '
        )
        # 改loss -> width
        if args.do_train_ssim:
            do_and_log_evaluation(args, epoch, recon_model, model, train_loader, writer, 'Train', train_data_range_dict)
        do_and_log_evaluation(args, epoch, recon_model, model, dev_loader, writer, 'Val', dev_data_range_dict)

        scheduler.step()
        save_policy_model(args, args.run_dir, epoch, model, optimiser)
    writer.close()


def save_policy_model(args, exp_dir, epoch, model, optimizer):
    torch.save(
        {
            'epoch': epoch,
            'args': args,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'exp_dir': exp_dir
        },
        f=exp_dir / 'model.pt'
    )

    if epoch in args.milestones:
        torch.save(
            {
                'epoch': epoch,
                'args': args,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'exp_dir': exp_dir
            },
            f=exp_dir / f'model_{epoch}.pt'
        )


def do_and_log_evaluation(args, epoch, recon_model, model, loader, writer, partition, data_range_dict):
    """
    Helper function for logging.
    """
    ssims, psnrs, score_time = evaluate(args, epoch, recon_model, model, loader, writer, partition, data_range_dict)
    ssims_str = ", ".join(["{}: {:.4f}".format(i, l) for i, l in enumerate(ssims)])
    psnrs_str = ", ".join(["{}: {:.3f}".format(i, l) for i, l in enumerate(psnrs)])
    logging.info(f'{partition}SSIM = [{ssims_str}]')
    logging.info(f'{partition}PSNR = [{psnrs_str}]')
    logging.info(f'{partition}ScoreTime = {score_time:.2f}s')


class PolicyNet(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(PolicyNet, self).__init__()
        self.outplanes = outplanes
        self.conv = nn.Conv2d(inplanes, 1, kernel_size=1)
        self.bn = nn.BatchNorm2d(1)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.fc = nn.Linear(outplanes - 1, outplanes)

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        x = x.view(-1, self.outplanes - 1)
        x = self.fc(x)
        probas = self.logsoftmax(x).exp()
        return probas


def evaluate(args, epoch, recon_model, model, loader, writer, partition, data_range_dict):
    """
    Evaluates the policy on all slices in a validation or test dataset on the SSIM and PSNR metrics.

    :param args: Argument object, containing hyperparameters for model evaluation.
    :param epoch: int, current training epoch.
    :param recon_model: reconstruction model object.
    :param model: policy model object.
    :param loader: training data loader.
    :param writer: Tensorboard writer.
    :param partition: str, dataset partition to evaluate on ('val' or 'test')
    :param data_range_dict: dictionary containing the dynamic range of every volume in the validation or test data.
    :return: (dict: average SSIMS per time step, dict: average PSNR per time step, float: evaluation duration)
    """
    model.eval()
    tbs = 0  # data set size counter
    start = time.perf_counter()
    with torch.no_grad():
        for it, data in enumerate(loader):
            kspace, masked_kspace, mask, zf, gt, gt_mean, gt_std, fname, _ = data
            # shape after unsqueeze = batch x channel x columns x rows x complex
            kspace = kspace.unsqueeze(1).to(args.device)
            masked_kspace = masked_kspace.unsqueeze(1).to(args.device)
            mask = mask.unsqueeze(1).to(args.device)
            # shape after unsqueeze = batch x channel x columns x rows
            zf = zf.unsqueeze(1).to(args.device)
            gt = gt.unsqueeze(1).to(args.device)
            gt_mean = gt_mean.unsqueeze(1).unsqueeze(2).unsqueeze(3).to(args.device)
            gt_std = gt_std.unsqueeze(1).unsqueeze(2).unsqueeze(3).to(args.device)
            unnorm_gt = gt * gt_std + gt_mean
            data_range = torch.stack([data_range_dict[vol] for vol in fname])
            tbs += mask.size(0)

            # Base reconstruction model forward pass
            recons = recon_model(zf)
            # 改ssim -> compute_score
            unnorm_recons = recons[:, :, :, :] * gt_std + gt_mean
            init_ssim_val = compute_ssim(unnorm_recons, unnorm_gt, size_average=False,
                                         data_range=data_range).mean(dim=(-1, -2)).sum()
            init_psnr_val = compute_psnr(args, unnorm_recons, unnorm_gt, data_range).sum()

            batch_ssims = [init_ssim_val.item()]
            batch_psnrs = [init_psnr_val.item()]

            for step in range(args.acquisition_steps):
                policy, probs = get_policy_probs(model, recons, mask)
                if step == 0:
                    actions = torch.multinomial(probs.squeeze(1), args.num_test_trajectories, replacement=True)
                else:
                    actions = policy.sample()
                # Samples trajectories in parallel
                # For evaluation we can treat greedy and non-greedy the same: in both cases we just simulate
                # num_test_trajectories acquisition trajectories in parallel for each slice in the batch, and store
                # the average SSIM score every time step.
                mask, masked_kspace, zf, recons = compute_next_step_reconstruction(recon_model, kspace,
                                                                                   masked_kspace, mask, actions)

                score = compute_scores(width, position)


    # Logging
    if partition in ['Val', 'Train']:
        for step, val in enumerate(ssims):
            writer.add_scalar(f'{partition}SSIM_step{step}', val, epoch)
            writer.add_scalar(f'{partition}PSNR_step{step}', psnrs[step], epoch)

        if args.wandb:
            wandb.log({f'{partition.lower()}_ssims': {str(key): val for key, val in enumerate(ssims)}}, step=epoch + 1)
            wandb.log({f'{partition.lower()}_psnrs': {str(key): val for key, val in enumerate(psnrs)}}, step=epoch + 1)

    elif partition == 'Test':
        # Only computed once, so loop over all epochs for wandb logging
        if args.wandb:
            for epoch in range(args.num_epochs):
                wandb.log({f'{partition.lower()}_ssims': {str(key): val for key, val in enumerate(ssims)}},
                          step=epoch + 1)
                wandb.log({f'{partition.lower()}_psnrs': {str(key): val for key, val in enumerate(psnrs)}},
                          step=epoch + 1)

    else:
        raise ValueError(f"'partition' should be in ['Train', 'Val', 'Test'], not: {partition}")

    return time.perf_counter() - start


def main(args):
    """
    Wrapper for training and testing of policy models.
    """
    logging.info(args)
    # Reconstruction model
    recon_args, recon_model = load_recon_model(w, pos)

    # Policy model to train
    # args.do_train -> default to be true
    if args.do_train:
        train_and_eval(args, recon_args, recon_model)
    else:
        test(args, recon_model)


def test(args, recon_model):
    """
    Performs evaluation of a pre-trained policy model.

    :param args: Argument object containing evaluation parameters.
    :param recon_model: reconstruction model.
    """
    model, policy_args = load_policy_model(pathlib.Path(args.policy_model_checkpoint))

    # Overwrite number of trajectories to test on
    policy_args.num_test_trajectories = args.num_test_trajectories
    if args.data_path is not None:  # Overwrite data path if provided
        policy_args.data_path = args.data_path

    # Logging of policy model
    logging.info(args)
    logging.info(recon_model)
    logging.info(model)
    if args.wandb:
        wandb.config.update(args)
        wandb.watch(model, log='all')
    # Initialise summary writer
    writer = SummaryWriter(log_dir=policy_args.run_dir / 'summary')

    # Parameter counting
    logging.info('Reconstruction model parameters: total {}, of which {} trainable and {} untrainable'.format(
        count_parameters(recon_model), count_trainable_parameters(recon_model),
        count_untrainable_parameters(recon_model)))
    logging.info('Policy model parameters: total {}, of which {} trainable and {} untrainable'.format(
        count_parameters(model), count_trainable_parameters(model), count_untrainable_parameters(model)))

    # Create data loader
    test_loader = create_data_loader(policy_args, 'test', shuffle=False)
    test_data_range_dict = create_data_range_dict(policy_args, test_loader)

    do_and_log_evaluation(policy_args, -1, recon_model, model, test_loader, writer, 'Test', test_data_range_dict)

    writer.close()


def wrap_main(args):
    """
    Wrapper for the entire script. Performs some setup, such as setting seed and starting wandb.
    """
    if args.seed != 0:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.device == 'cuda':
            torch.cuda.manual_seed(args.seed)

    args.milestones = args.milestones + [0, args.num_epochs - 1]

    if args.wandb:
        if args.resume:
            assert args.run_id is not None, "run_id must be given if resuming with wandb."
            wandb.init(project=args.project, resume=args.run_id)
        elif args.test_multi:
            wandb.init(project=args.project, reinit=True)
        else:
            wandb.init(project=args.project, config=args)

    # To get reproducible behaviour, additionally set args.num_workers = 0 and disable cudnn
    # torch.backends.cudnn.enabled = False

    main(args)


if __name__ == "__main__":

    w = 10
    pos = 60
    kspace = compute_kspace(w)
    mask = compute_mask(w, pos)
    masked_kspace = compute_masked_kspace(kspace, mask)
    recon = load_recon_model(w, pos)
    score = compute_scores(recon)
    print(score)

    # n = 4096
    # r = 256
    # w = 20data_loading.py
    # offset = 256
    # dispRange: List[Union[int, float]] = []
    # for i in range(-600, 601):
    #     dispRange.append(i + math.floor(n / 2) + 1)
    # v = []
    # for i in range(0, n):
    #     v.append(i - math.floor(n / 2))
    # annulus = createAnnulus(v, r, w)
    #
    # initial_v = []
    # for i in range(0, n):
    #     initial_v.append(-v[i])
    # selected_columns = []
    # for i in range(0, n):
    #     if offset * 0.99 / 1.35 + w / 2 > initial_v[i] > offset * 0.99 / 1.35 - w / 2:
    #         selected_columns.append(1)
    #     else:
    #         selected_columns.append(0)

    # step 1: specify the initial v position
    # n = 4096
    # r = 256
    # w = 10
    # pos = 60
    # offset = 256
    # dispRange: List[Union[int, float]] = []
    # for i in range(-600, 601):
    #     dispRange.append(i + math.floor(n / 2) + 1)
    # v = []
    # for i in range(0, n):
    #     v.append(i - math.floor(n / 2))
    # annulus = createAnnulus(v, r, w)
    #
    # # step 2: start to move apart in two directions
    # initial_v = []
    # for i in range(0, n):
    #     initial_v.append(-v[i])
    # selected_columns = []
    # for i in range(0, n):
    #     if (offset * 0.99 / 1.35 + pos + w / 2 > initial_v[i] > offset * 0.99 / 1.35 + pos - w / 2) or \
    #             (offset * 0.99 / 1.35 - pos + w / 2 > initial_v[i] > offset * 0.99 / 1.35 - pos - w / 2):
    #         selected_columns.append(1)
    #     else:
    #         selected_columns.append(0)
    #
    # latticeFourierMask = annulus
    # for c in range(0, n):
    #     if not selected_columns[c]:
    #         latticeFourierMask[:, c] = False
    # latticeFourierMask = latticeFourierMask.astype(float)
    #
    # fieldSynthesisProfile = ft.fftshift(ft.ifft(ft.ifftshift(latticeFourierMask)))
    # df = pd.DataFrame(fieldSynthesisProfile)
    # sum_rows = df.sum(axis=1)
    # fieldSynthesisProfile = sum_rows ** 2
    # lattice_efield = ft.fftshift(ft.ifft2(ft.ifftshift(latticeFourierMask)))
    # lattice = abs(lattice_efield) ** 2
    # df1 = pd.DataFrame(lattice)
    # latticeLineProfile = df1.sum(axis=1)
    # latticeLineProfile = latticeLineProfile * n
    #
    # lattice_hat = ft.fftshift(ft.fft2(ft.ifftshift(lattice)))
    #
    # period = n / offset
    # if period == round(period):
    #     latticeDithered = conv2(lattice, np.ones((1, int(period))) / period, 'same')
    # else:
    #     for x in lattice_hat:
    #         for y in x:
    #             latticeDithered = y * (np.sinc(v / period))
    #     latticeDithered = ft.fftshift(ft.ifft2(ft.ifftshift(latticeDithered)))
    #
    # latticeDithered_hat = ft.fftshift(ft.fft2(ft.ifftshift(latticeDithered)))
    # width = osWidth_gui_2(latticeDithered)
    # print(width)

    # # Gaussian Beam Test
    # x = np.arange(-3.1, 3.2, 0.1)
    # # Mean = 0, SD = 1.
    # mean = 0
    # std = 1
    # variance = np.square(std)
    # prof_peak = np.exp(-np.square(x - mean) / 2 * variance) / (np.sqrt(2 * np.pi * variance))
    # plt.plot(x, prof_peak)
    # w = osWidth_gui(prof_peak)

    # Draw the light sheet
    # fig = plt.figure()
    # plt.figure(figsize=(16, 9))
    # plt.gca().get_autoscale_on()
    #
    # ax1 = plt.subplot(231)
    # ax1.set_xlim([1500, 2500])
    # ax1.set_ylim([1500, 2500])
    # aa = abs(latticeFourierMask)
    # aa[aa > 1e-6] = 1e-6
    # ax1.imshow(latticeFourierMask, cmap='hot')
    # ax1.set_title('Electric field in pupil')
    #
    # aaa = abs(lattice_hat)
    # aaa[aaa > 1e-6] = 1e-6
    # ax2 = plt.subplot(232)
    # ax2.set_xlim([1250, 2800])
    # ax2.set_ylim([1250, 2800])
    # ax2.imshow(abs(aaa), cmap='hot')
    # ax2.set_title('Fourier components of lattice intensity')
    #
    # bbb = abs(latticeDithered_hat)
    # bbb[bbb > 1e-6] = 1e-6
    # ax3 = plt.subplot(233)
    # ax3.set_xlim([1300, 2800])
    # ax3.set_ylim([1300, 2800])
    # ax3.imshow(abs(bbb), cmap='hot')
    # ax3.set_title('Fourier components of dithered lattice intensity')
    #
    # ax4 = plt.subplot(234)
    # ax4.imshow(np.real(lattice_efield), cmap='gray')
    # ax4.set_title('Electric field of lattice at focal plane')
    # ax4.set_xlim([1950, 2150])
    # ax4.set_ylim([1950, 2150])
    #
    # ax5 = plt.subplot(235)
    # ax5.imshow(lattice, cmap='hot')
    # ax5.set_title('Intensity of lattice')
    # ax5.set_xlim([1950, 2150])
    # ax5.set_ylim([1950, 2150])
    #
    # ax6 = plt.subplot(236)
    # ax6.imshow(latticeDithered, cmap='hot')
    # ax6.set_title('Averaged Intensity of dithered lattice')
    # ax6.set_xlim([1950, 2150])
    # ax6.set_ylim([1950, 2150])
    #
    # plt.show()
    # plt.pause(0.001)

    ##demoFieldSynthesis()

    # model, = load_policy_model()
    # policy, probs = get_policy_probs(model, recon, mask)

    # import torch.multiprocessing
    # torch.multiprocessing.set_start_method('spawn')
    #
    # base_args = create_arg_parser().parse_args()
    #
    # # Testing multiple policy models with one script
    # if base_args.test_multi:
    #     assert not base_args.do_train, "Doing multiple model testing: do_train must be False."
    #     assert base_args.policy_model_list[0] is not None, ("Doing multiple model testing: must "
    #                                                         "have list of policy models.")
    #
    #     for model in base_args.policy_model_list:
    #         args = copy.deepcopy(base_args)
    #         args.policy_model_checkpoint = model
    #         wrap_main(args)
    #         wandb.join()
    #
    # else:
    #     wrap_main(base_args)
