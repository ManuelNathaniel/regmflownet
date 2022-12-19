from __future__ import print_function, division
import sys

import argparse, configparser
import os
# import cv2
import time
import numpy as np
import math
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import datetime

from torch.utils.data import DataLoader
from torch.optim.adamw import AdamW
from core.onecyclelr import OneCycleLR
from core import create_model

from core.loss import compute_supervision_coarse, compute_coarse_loss, backwarp

import evaluate
import core.datasets as datasets

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DOP

from tensorboardX import SummaryWriter

sys.path.append('core')

try:
    from torch.cuda.amp import GradScaler
except:
    # dummy GradScaler for PyTorch < 1.6
    class GradScaler:
        def __init__(self, enabled=False):
            pass

        def scale(self, loss):
            return loss

        def unscale_(self, optimizer):
            pass

        def step(self, optimizer):
            optimizer.step()

        def update(self):
            pass

# exclude extremely large displacements
MAX_FLOW = 400
SUM_FREQ = 100
VAL_FREQ = 100


def sequence_loss(train_outputs, image1, image2, flow_gt, valid, gamma=0.8, max_flow=MAX_FLOW, use_matching_loss=False):
    """ Loss function defined over sequence of flow predictions """
    flow_preds, softCorrMap = train_outputs

    # original RAFT loss
    n_predictions = len(flow_preds)
    flow_loss = 0.0

    # exclude invalid pixels and extremely large displacements
    mag = torch.sum(flow_gt ** 2, dim=1).sqrt()
    valid = (valid >= 0.5) & (mag < max_flow)

    for i in range(n_predictions):
        i_weight = gamma ** (n_predictions - i - 1)
        i_loss = (flow_preds[i] - flow_gt).abs()
        flow_loss += i_weight * (valid[:, None].float() * i_loss).mean()

    epe = torch.sum((flow_preds[-1] - flow_gt) ** 2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }

    if use_matching_loss:
        # enable global matching loss. Try to use it in late stages of the trianing
        img_2back1 = backwarp(image2, flow_gt)
        occlusionMap = (image1 - img_2back1).mean(1, keepdims=True)  # (N, H, W)
        occlusionMap = torch.abs(occlusionMap) > 20
        occlusionMap = occlusionMap.float()

        conf_matrix_gt = compute_supervision_coarse(flow_gt, occlusionMap, 8)  # 8 from RAFT downsample

        matchLossCfg = configparser.ConfigParser()
        matchLossCfg.POS_WEIGHT = 1
        matchLossCfg.NEG_WEIGHT = 1
        matchLossCfg.FOCAL_ALPHA = 0.25
        matchLossCfg.FOCAL_GAMMA = 2.0
        matchLossCfg.COARSE_TYPE = 'cross_entropy'
        match_loss = compute_coarse_loss(softCorrMap, conf_matrix_gt, matchLossCfg)

        flow_loss = flow_loss + 0.01 * match_loss

    return flow_loss, metrics


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def fetch_optimizer(args, model, steps_per_epoch, last_iters=-1):
    """ Create the optimizer and learning rate scheduler """
    optimizer = AdamW(model.parameters(), lr=args.max_lr, weight_decay=args.wdecay, eps=args.epsilon)

    scheduler = OneCycleLR(optimizer, max_lr=args.max_lr, epochs=args.epochs, steps_per_epoch=steps_per_epoch,
                           pct_start=args.pct_start, anneal_strategy=args.anneal_strategy,
                           cycle_momentum=args.cycle_momentum, base_momentum=0.85, max_momentum=0.95, last_epoch=last_iters)

    return optimizer, scheduler


class Logger:
    def __init__(self, model, scheduler, total_steps=0, log_dir=None):
        self.model = model
        self.scheduler = scheduler
        self.total_steps = total_steps
        self.running_loss = {}
        self.writer = None
        self.log_dir = log_dir

    def _print_training_status(self):
        metrics_data = [self.running_loss[k] / SUM_FREQ for k in sorted(self.running_loss.keys())]
        if not hasattr(self.scheduler, 'get_last_lr'):
            training_str = "[{:6d}] ".format(self.total_steps + 1)
        else:
            training_str = "{:6d}, {:10.7f}] ".format(self.total_steps + 1,
                                                              self.scheduler.get_last_lr()[0])
        metrics_str = ("{:10.4f}, " * len(metrics_data)).format(*metrics_data)

        curTime = datetime.datetime.now()
        time_str = '{}: '.format(curTime.strftime('%Y-%m-%d %H:%M:%S'))
        # print the training status
        print(time_str + training_str + metrics_str)

        if self.writer is None:
            self.writer = SummaryWriter(logdir=self.log_dir)

        for k in self.running_loss:
            self.writer.add_scalar(k, self.running_loss[k] / SUM_FREQ, self.total_steps)
            self.running_loss[k] = 0.0

    def push(self, metrics):
        self.total_steps += 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % SUM_FREQ == SUM_FREQ - 1:
            self._print_training_status()
            self.running_loss = {}

    def write_dict(self, results):
        if self.writer is None:
            self.writer = SummaryWriter(logdir=self.log_dir)

        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps)

    def close(self):
        self.writer.close()


def write_msg(paras=None, msg=None, pr=True, p_end='\n', flush=False, p_sep=' ', wr=True, add=True, w_end='\n', disptime=True):
    curTime = datetime.datetime.now()
    time_str = curTime.strftime('%Y-%m-%d %H:%M:%S')
    if disptime:
        time_str = '[{}]'.format(time_str)
    else:
        time_str = ' '
        
    if pr:
        # if print
        print('{} \t {}'.format(time_str, msg), sep=p_sep, end=p_end, flush=flush)

    if not wr:
        # if not write msg to file
        return False

    fpt = "%s/%s.txt" % (paras.logs, paras.log_name)
    if not os.path.exists(fpt):
        mod = 'w'
    else:
        if add:
            mod = 'a+'  # 追加模式
        else:
            mod = 'w'  # 覆写模式

    f = open(fpt, mod)
    f.write("{} \t {}{}".format(time_str, msg, w_end))
    f.close()


def file_manager(paras):
    """文件管理器：管理训练过程中文件的存储
    """
    if not os.path.exists(paras.code_version):
        os.mkdir(paras.code_version)

    paras.checkpoints = '%s/%s' % (paras.code_version, 'checkpoints')
    paras.runs = '%s/%s' % (paras.code_version, 'runs')
    paras.logs = '%s/%s' % (paras.code_version, 'logs')

    dir_list = [paras.checkpoints, paras.runs, paras.logs]
    for dir in dir_list:
        if not os.path.exists(dir):
            os.mkdir(dir)

    return paras


def train(args):
    model = nn.DataParallel(create_model(args), device_ids=args.gpus)
    info_stream = "Parameter Count: %d" % count_parameters(model)

    if args.restore_ckpt is not None:
        model.load_state_dict(torch.load(args.restore_ckpt), strict=True)
        write_msg(args, msg=info_stream, add=True)
    else:
        write_msg(args, msg=info_stream, add=False)

    model.cuda()
    model.train()

    if args.freeze_bn:
        model.module.freeze_bn()

    if args.restore_ckpt is not None:
        strStep = os.path.split(args.restore_ckpt)[-1].split('_')[0]
        total_steps = int(strStep) if strStep.isdigit() else 0
    else:
        total_steps = 0

    train_loader, _ = datasets.fetch_dataloader(args, TRAIN_DS='B+C+D+J+S+U')
    steps_per_epoch = int(len(train_loader))
    optimizer, scheduler = fetch_optimizer(args, model, steps_per_epoch=steps_per_epoch, last_iters=total_steps)

    scaler = GradScaler(enabled=args.mixed_precision)
    logger = Logger(model, scheduler, total_steps, os.path.join(args.runs, args.name))

    # ************************* Training Begin *************************
    info_stream = '\n{0:*^140}'.format(' Training Begin ')
    write_msg(args, msg=info_stream, disptime=False)

    lowest_epe_list = []
    lowest_epe = 10000.0
    for epoch in range(args.epochs):
        for i_batch, data_blob in enumerate(train_loader):
            optimizer.zero_grad()
            image1, image2, flow, valid = [x.cuda() for x in data_blob]

            if args.add_noise:
                stdv = np.random.uniform(0.0, 5.0)
                image1 = (image1 + stdv * torch.randn(*image1.shape).cuda()).clamp(0.0, 255.0)
                image2 = (image2 + stdv * torch.randn(*image2.shape).cuda()).clamp(0.0, 255.0)

            flow_predictions = model(image1, image2, iters=args.iters)
            loss, metrics = sequence_loss(flow_predictions, image1, image2, flow, valid, gamma=args.gamma,
                                          use_matching_loss=args.use_mix_attn)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

            total_steps += 1
            # logger.push(metrics)

            # >>>>>>>>>>>>>>>>>>> Train Epoch={} End, Validation & Continue >>>>>>>>>>>>>>>>>>> 
            train_info = 'Train Epoch = {:5d}, Steps/Batch = {:6d}, Batch EPE: {:10.6f}, 1px: {:10.6f}, 3px: {:10.6f},' \
                         ' 5px: {:10.6f}, Lr: {:10.6f}'.format(epoch+1, total_steps, metrics['epe'], metrics['1px'], metrics['3px'],
                                                 metrics['5px'], scheduler.get_last_lr()[0])
            write_msg(args, msg=train_info, disptime=True)

            ## 每100个batch验证一次
            if total_steps % VAL_FREQ == VAL_FREQ - 1:
                write_msg(args, msg='To Be Validate: ', p_end='', w_end='', flush=True)
                stage_dir = '%s/%s' % (args.checkpoints, args.stage)
                if not os.path.exists(stage_dir):
                    os.mkdir(stage_dir)
                PATH = f'{stage_dir}/{args.model}_{args.stage}_{args.flowtype}_{args.test}_{args.sn}_{total_steps:d}.pth'

                results = {}
                for val_dataset in args.validation:
                    r1 = evaluate.validate_baseflow(model.module, paras=args, split='validate', flowtype=val_dataset)
                    r2 = evaluate.test_baseflow(model.module, paras=args, split='test', flowtype=val_dataset)
                    write_msg(args, msg='Validate EPE: {:10.6f}, Lr: {:10.6f}'.format(r1[val_dataset], scheduler.get_last_lr()[0]), disptime=False)
                    write_msg(args, msg='Test EPE: {:10.6f}'.format(r2[val_dataset]))
                    if r1[val_dataset] < lowest_epe:
                        lowest_epe = r1[val_dataset]
                        torch.save(model.state_dict(), PATH)
                        write_msg(args, msg='model saved and lowest epe overwritten: {:10.6f}'.format(lowest_epe), flush=True, disptime=True)
                        lowest_epe_list.append(r1[val_dataset])
                        
                write_msg(args, msg='\n{}'.format('-' * 60), disptime=False)
                # logger.write_dict(results)

                model.train()
                if args.freeze_bn:
                    model.module.freeze_bn()
        torch.cuda.empty_cache()

    logger.close()
    PATH = '%s/%s.pth' % (args.checkpoints, args.name)
    torch.save(model.state_dict(), PATH)

    return PATH


def train_epochs(args):
    """
    按给定epochs进行训练，不用总steps进行学习率下降的训练
    """
    model = nn.DataParallel(create_model(args), device_ids=args.gpus)
    info_stream = "Parameter Count: %d" % count_parameters(model)

    if args.restore_ckpt is not None:
        checkpoint = torch.load(args.restore_ckpt)
        model.load_state_dict(checkpoint, strict=True)
        write_msg(args, msg=info_stream, add=True)
    else:
        write_msg(args, msg=info_stream, add=False)

    model.cuda()
    model.train()

    if args.freeze_bn:
        model.module.freeze_bn()

    train_loader, validate_loader = datasets.fetch_dataloader(args, TRAIN_DS='C+T+K/S')
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.init_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=args.reduce_factor,
                                                           patience=args.patience_level, min_lr=args.min_lr)
    scaler = GradScaler(enabled=args.mixed_precision)

    if args.restore_ckpt is not None:
        strStep = os.path.split(args.restore_ckpt)[-1].split('_')[1]
        strEpoch = os.path.split(args.restore_ckpt)[-1].split('_')[0]
        total_steps = int(strStep) if strStep.isdigit() else 0
        total_epochs = int(strEpoch) if strEpoch.isdigit() else 0
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    else:
        total_epochs = 0
        total_steps = 0
    # logger = Logger(model, scheduler, total_steps=total_steps, log_dir=os.path.join(args.runs, args.name),
    #                 total_epochs=total_epochs)

    # ************************* Training Begin *************************
    info_stream = '\n{0:*^140}'.format(' Training Begin ')
    write_msg(args, msg=info_stream)

    lowest_validation_epe = 100000.0
    # Training
    for epoch in range(args.epochs):
        sum_train_loss, sum_validate_loss = 0.0, 0.0
        total_train_samples, total_validate_samples = 0, 0
        sum_train_epe, sum_validate_epe = 0.0, 0.0

        info_stream = '{0:-^60}'.format('Train Epoch = ' + str(epoch+1))
        write_msg(args, msg=info_stream)

        for i_batch, data_blob in enumerate(train_loader):
            optimizer.zero_grad()
            image1, image2, flow, valid = [x.cuda() for x in data_blob]

            if args.add_noise:
                stdv = np.random.uniform(0.0, 5.0)
                image1 = (image1 + stdv * torch.randn(*image1.shape).cuda()).clamp(0.0, 255.0)
                image2 = (image2 + stdv * torch.randn(*image2.shape).cuda()).clamp(0.0, 255.0)

            flow_predictions = model(image1, image2, iters=args.iters)
            training_loss, metrics = sequence_loss(flow_predictions, image1, image2, flow, valid, gamma=args.gamma,
                                                   use_matching_loss=args.use_mix_attn)

            sum_train_loss += training_loss.item() * image1.shape[0]
            total_train_samples += image1.shape[0]
            epoch_train_loss = sum_train_loss / total_train_samples
            sum_train_epe += metrics['epe'] * image1.shape[0]
            epoch_train_epe_loss = sum_train_epe / total_train_samples

            scaler.scale(training_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

            scaler.step(optimizer)
            scaler.update()

            train_pinfo = '[{}]: Epochs: {:4d}, Steps: {:5d}, Sum Loss: {:10.5f}, Total Samples: {:6d}, Sum Epe: ' \
                          '{:10.5f}, Epoch Loss: {:10.5f}, Epoch EPE: {:10.5f}, 1px: {:10.6f}, 3px:{:10.6f}, ' \
                          '5px: {:10.6f}' \
                .format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), epoch+1, total_steps+1, sum_train_loss,
                        total_train_samples, sum_train_epe, epoch_train_loss, epoch_train_epe_loss, metrics['1px'],
                        metrics['3px'], metrics['5px'])
            write_msg(args, msg=train_pinfo, flush=True)
            # logger.push(metrics=metrics, total_epochs=epoch+1)
            total_steps += 1

        # >>>>>>>>>>>>>>>>>>> Train Epoch={} End, Validation & Continue >>>>>>>>>>>>>>>>>>>
        info_stream = '{0:-^60}'.format('Train Epoch = ' + str(epoch + 1) + ' To Be Validate')
        write_msg(args, msg=info_stream)
        with torch.set_grad_enabled(False):
            model.eval()

            stage_dir = '%s/%s' % (args.checkpoints, args.stage)
            if not os.path.exists(stage_dir):
                os.mkdir(stage_dir)

            # >>>>>>>>>> GMFlowNet的计算方法 >>>>>>>>>>
            write_msg(args, msg='----------> GMFlowNet EPE: ', p_end="", w_end="")
            results = {}
            for validate_dataset_name in args.validation:
                results_1 = evaluate.validate_baseflow(model.module, paras=args, split='validate', flowtype=validate_dataset_name)
                results.update(results_1)

                info_stream = '{}: {:10.6f}'.format(validate_dataset_name, results_1[validate_dataset_name])
                write_msg(args, msg=info_stream)
            # <<<<<<<<<<< END <<<<<<<<<

            # >>>>>>>>> DORFlowNet中的计算思路 >>>>>>>>>>>
            write_msg(args, msg='----------> DROFlowNet: ', p_end='', w_end='')
            drof_epe_list = []
            for i, sample_batched in enumerate(validate_loader):
                val_img1, val_img2, val_flow_gt, val_valid = [x.cuda() for x in sample_batched]
                val_flow_pr, _ = model(val_img1, val_img2, iters=24)
                n_predictions = len(val_flow_pr)
                flow_loss = 0.0
                for n in range(n_predictions):
                    n_weight = 0.8 ** (n_predictions - n - 1)
                    n_loss = (val_flow_pr[n] - val_flow_gt).abs()
                    flow_loss += n_weight * n_loss.mean()
                    
                validate_epe_loss_one_batch = torch.sum((val_flow_pr[-1] - val_flow_gt) ** 2, dim=1).sqrt()
                validate_epe_loss = validate_epe_loss_one_batch.view(-1).mean().item()

                validate_loss = flow_loss
                sum_validate_loss += validate_loss.item() * val_img1.shape[0]
                total_validate_samples += val_img1.shape[0]
                epoch_validate_loss = sum_validate_loss / total_validate_samples

                sum_validate_epe += validate_epe_loss * val_img1.shape[0]
                epoch_validate_epe_loss = sum_validate_epe / total_validate_samples

                drof_epe_list.append(validate_epe_loss)
            drof_epe = np.mean(drof_epe_list)
            # <<<<<<<<<< END <<<<<<<<<<

            info_stream = 'Loss: {:10.6f}, EPE: {:10.6f}, '.format(epoch_validate_loss, epoch_validate_epe_loss)
            write_msg(args, msg=info_stream, p_end='', w_end='')
            write_msg(args, msg='EPE Mean: {:10.6f}'.format(drof_epe))
            # logger.write_dict(results)
            
        scheduler.step(drof_epe)

        write_msg(args, msg='---------->{}'.format(' Validation End & Summarize '), p_end='', w_end='')
        info_stream = '\tLr: {:10.6f}'.format(optimizer.__getattribute__('param_groups')[0]['lr'])
        write_msg(args, msg=info_stream, flush=True)
        
        if drof_epe < lowest_validation_epe:
            lowest_validation_epe = drof_epe
            PATH = f'{stage_dir}/{args.model}_{args.stage}_{args.flowtype}_{args.test}_{args.sn}_{epoch+1}_{total_steps + 1:d}.pth '
            torch.save({
                'epoch':epoch,
                'model_state_dice': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict()
            }, PATH)
            write_msg(args, msg='model saved and lowest epe overwritten: {:10.6f}'.format(lowest_validation_epe), flush=True)
        write_msg(args, msg='{}\n'.format('-'*60))

        torch.cuda.empty_cache()
        model.train()
        if args.freeze_bn:
            model.module.freeze_bn()

    # logger.close()
    PATH = '%s/%s.pth' % (args.checkpoints, args.name)
    torch.save(model.state_dict(), PATH)
    # return PATH


def params_config():
    flow_list = ['backstep', 'cylinder', 'DNS_turbulence', 'JHTDB_channel', 'JHTDB_channel_hd',
                 'JHTDB_isotropic_1024_hd', 'JHTDB_mhd1024_hd', 'SQG', 'uniform']

    parser = argparse.ArgumentParser()

    # Code version
    parser.add_argument('--code_version', default='v1.2.10', help="program version number")
    parser.add_argument('-m', '--method', default=None, choices=['epoch', 'step'])

    # Net architecture
    parser.add_argument('--name', default=None,
                        help="name of your experiment. The saved checkpoint will be named after this in `./checkpoints/.`")
    parser.add_argument('--model', default='gmflownet',
                        help="mdoel class. `<args.model>`_model.py should be in ./core and `<args.model>Model` should be defined in this file")
    parser.add_argument('--use_mix_attn', action='store_true', help='use mixture of POLA and axial attentions')
    parser.add_argument('--freeze_bn', action='store_true')
    parser.add_argument('--small', action="store_true")

    # datasets parameters
    parser.add_argument('--train_model', default='single', choices=['single', 'combine'])
    parser.add_argument('--test', default=None, help='Number of experiments identifier, such as T1')
    parser.add_argument('--sn', type=str, default=None, help="yyyymmddhh(年月日时训练的版本，不是程序的版本号)")
    parser.add_argument('--stage', default='baseflow', help="determines which dataset to use for training")
    parser.add_argument('--flowtype', default=None, type=str)
    parser.add_argument('--validation', default=None, type=str, nargs='+')
    parser.add_argument('--restore_ckpt', default=None, help="restore checkpoint")
    parser.add_argument('--flow_list', type=list, default=flow_list)

    # image parameters
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--image_size', type=int, nargs='+', default=[224, 224])
    parser.add_argument('--add_noise', action='store_true')
    parser.add_argument('--gpus', type=int, nargs='+', default=[0])
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')

    # learning parameters by steps
    parser.add_argument('--max_lr', type=float, default=0.0006)
    parser.add_argument('--pct_start', type=float, default=0.05)
    parser.add_argument('--anneal_strategy', default='linear', choices=['cos', 'linear'])
    parser.add_argument('--cycle_momentum', action='store_true')
    parser.add_argument('--num_steps', type=int, default=100000)
    parser.add_argument('--iters', type=int, default=12)
    parser.add_argument('--wdecay', type=float, default=.0005)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--gamma', type=float, default=0.8, help='exponential weighting')

    # learning parameters by ReduceLROnPlateau scheme
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--init_lr', type=float, default=0.0002)
    parser.add_argument('--reduce_factor', type=float, default=0.2)
    parser.add_argument('--patience_level', type=int, default=5)
    parser.add_argument('--min_lr', type=float, default=1e-8)

    # files manage
    parser.add_argument('--log_name', default=None)
    parser.add_argument('--checkpoints', default='v1.0.0/checkpoints')
    parser.add_argument('--runs', default='v1.0.0/runs')
    parser.add_argument('--logs', default='v1.0.0/logs')
    parser.add_argument('--client', default='server', choices=['local', 'server'], help='operating environment')
    parser.add_argument('--data_path', default=None, help='Datasets storage path')
    parser.add_argument('--local', default='D:/Datasets/baseflow', help='local detailed path')
    parser.add_argument('--server', default='/home/dsm/flownet/datasets/baseflow', help='server detailed path')

    args = parser.parse_args()

    if args.sn is None:
        args.sn = datetime.datetime.now().strftime('%Y%m%d%H')

    args.flow_list = flow_list
    args = file_manager(args)
    client_path = {'local': args.local, 'server': args.server}
    args.data_path = client_path[args.client]
    args.name = '{}_{}_{}_{}_{}'.format(args.model, args.stage, args.flowtype, args.method, args.test)
    args.log_name = '{}_{}'.format(args.name, args.sn)

    return args


if __name__ == '__main__':
    args = params_config()

    torch.set_num_threads(16)
    torch.manual_seed(1234)
    np.random.seed(1234)

    if not os.path.exists('checkpoints'):
        os.mkdir('checkpoints')
    if not os.path.exists('runs'):
        os.mkdir('runs')

    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, args.gpus))
    args.gpus = [i for i in range(len(args.gpus))]

    if args.method == 'epoch':
        train_epochs(args)
    else:
        train(args)