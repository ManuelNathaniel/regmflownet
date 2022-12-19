import sys

from PIL import Image
import argparse
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

import core.datasets as datasets
from core.utils import flow_viz
from core.utils import frame_utils

# from raft import RAFT, RAFT_Transformer
from core import create_model
from core.utils.utils import InputPadder, forward_interpolate

sys.path.append('core')

@torch.no_grad()
def validate_baseflow(model, paras=None, iters=24, split=None, flowtype=None):
    """ Perform evaluation on the BaseFlow (test) split """
    model.eval()
    epe_list = []

    val_dataset = datasets.BaseFlow(split=split, is_validate=True, flowtype=flowtype)
    for val_id in range(len(val_dataset)):
        # validate过程的函数返回值有5个 img1 img2 flow valid extro_info
        image1, image2, flow_gt, _, _ = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        epe = torch.sum((flow_pr[0].cpu() - flow_gt) ** 2, dim=0).sqrt()
        epe_list.append(epe.view(-1).numpy())

    epe = np.mean(np.concatenate(epe_list))
    return {flowtype: epe}


@torch.no_grad()
def test_baseflow(model, paras=None, iters=24, split=None, flowtype=None):
    model.eval()
    epe_list = []
    test_dataset = datasets.BaseFlow(split=split, is_validate=False, flowtype=flowtype)

    for test_id in range(len(test_dataset)):
        image1, image2, flow_gt, _ = test_dataset[test_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()
        _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        epe = torch.sum((flow_pr[0].cpu() - flow_gt) ** 2, dim=0).sqrt()
        epe_list.append(epe.view(-1).numpy())

    epe = np.mean(np.concatenate(epe_list))
    return {flowtype: epe}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='gmflownet', help="mdoel class. `<args.model>`_model.py should be in ./core and `<args.model>Model` should be defined in this file")
    parser.add_argument('--ckpt', help="restored checkpoint")

    parser.add_argument('--dataset', help="dataset for evaluation")
    parser.add_argument('--use_mix_attn', action='store_true', help='use mixture of POLA and axial attentions')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    model = torch.nn.DataParallel(create_model(args))
    model.load_state_dict(torch.load(args.ckpt), strict=True)

    model.cuda()
    model.eval()

    with torch.no_grad():
        if args.dataset == 'baseflow':
            validate_baseflow(model)

