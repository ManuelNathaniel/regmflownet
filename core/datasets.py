# Data loading based on https://github.com/NVIDIA/flownet2-pytorch

import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F

import os
import math
import random
from glob import glob
import os.path as osp

from core.utils import frame_utils
from core.utils.augmentor import FlowAugmentor, SparseFlowAugmentor

import sys


class FlowDataset(data.Dataset):
    def __init__(self, aug_params=None, sparse=False):
        self.augmentor = None
        self.sparse = sparse
        if aug_params is not None:
            if sparse:
                self.augmentor = SparseFlowAugmentor(**aug_params)
            else:
                self.augmentor = FlowAugmentor(**aug_params)

        self.is_test = False
        self.is_validate = False
        self.init_seed = False
        self.flow_list = []
        self.image_list = []
        self.extra_info = []

    def __getitem__(self, index):
        # print('Index is {}'.format(index))
        # sys.stdout.flush()
        if self.is_test:
            # 同一个索引值存储这一对图像的路径
            img1 = frame_utils.read_gen(self.image_list[index][0])
            img2 = frame_utils.read_gen(self.image_list[index][1])

            img1 = np.array(img1).astype(np.uint8)
            img2 = np.array(img2).astype(np.uint8)
            # 把img除最后一个维度选前3维，由int32转换为uint8位类型
            if len(img1.shape) == 2:
                img1 = np.tile(img1[..., None], (1, 1, 3))
                img2 = np.tile(img2[..., None], (1, 1, 3))
            else:
                img1 = img1[..., :3]
                img2 = img2[..., :3]
            # 将img转换为tensor，并将维度交换成 C*H*W，转换为float型
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            flow = frame_utils.read_gen(self.flow_list[index])
            flow = torch.from_numpy(flow).permute(2, 0, 1).float()
            return img1, img2, flow, self.extra_info[index]

        index = index % len(self.image_list)
        valid = None
        if self.sparse:
            flow, valid = frame_utils.readFlowKITTI(self.flow_list[index])
        else:
            flow = frame_utils.read_gen(self.flow_list[index])

        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])

        flow = np.array(flow).astype(np.float32)
        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)

        # grayscale images
        if len(img1.shape) == 2:
            img1 = np.tile(img1[..., None], (1, 1, 3))
            img2 = np.tile(img2[..., None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]

        if self.augmentor is not None:
            if self.sparse:
                img1, img2, flow, valid = self.augmentor(img1, img2, flow, valid)
            else:
                img1, img2, flow = self.augmentor(img1, img2, flow)

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()

        if valid is not None:
            valid = torch.from_numpy(valid)
        else:
            valid = (flow[0].abs() < 1000) & (flow[1].abs() < 1000)

        if self.is_validate:
            return img1, img2, flow, valid.float(), self.extra_info[index]
        else:
            return img1, img2, flow, valid.float()

    def getDataWithPath(self, index):
        img1, img2, flow, valid = self.__getitem__(index)

        imgPath_1 = self.image_list[index][0]
        imgPath_2 = self.image_list[index][1]

        return img1, img2, flow, valid, imgPath_1, imgPath_2

    def __rmul__(self, v):
        self.flow_list = v * self.flow_list
        self.image_list = v * self.image_list
        return self

    def __len__(self):
        return len(self.image_list)


class BaseFlow(FlowDataset):
    def __init__(self, aug_params=None, split='train', root='/home/dsm/flownet/datasets/baseflow', is_validate=False,
                 flowtype=None):
        super(BaseFlow, self).__init__(aug_params)

        if flowtype is not None:
            scene = flowtype
            flow_scene_root = osp.join(root, split, scene, 'flow')
            image_scene_root = osp.join(root, split, scene, 'image')

            flow_scene_list = sorted(glob(osp.join(flow_scene_root, '*.flo')))
            image_scene_list = sorted(glob(osp.join(image_scene_root, '*.jpg')))

            for i in range(int(len(image_scene_list) / 2)):
                self.image_list += [[image_scene_list[2 * i], image_scene_list[2 * i + 1]]]
                self.extra_info += [(scene, i)]
            self.flow_list = flow_scene_list

        else:
            for scene in os.listdir(root):
                flow_scene_root = osp.join(root, split, scene, 'flow')
                image_scene_root = osp.join(root, split, scene, 'image')

                image_scene_list = sorted(glob(osp.join(image_scene_root, '*.jpg')))
                for i in range(int(len(image_scene_list) / 2)):
                    self.image_list += [[image_scene_list[2 * i], image_scene_list[2 * i + 1]]]
                    self.extra_info += [(scene, i)]
                self.flow_list += sorted(glob(osp.join(flow_scene_root, '*.flo')))

        self.is_validate = is_validate
        if split == 'test':
            # 是否是评估模式
            self.is_test = True
            # self.flow_list += sorted(glob(osp.join(flow_scene_root, '*.flo')))


def fetch_dataloader(args, TRAIN_DS='B+C+D/J'):
    """ Create the data loader for the corresponding trainign set """
    if args.stage == 'baseflow':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.1, 'max_scale': 1.0, 'do_flip': True}
        train_dataset = BaseFlow(aug_params=aug_params, split='train', flowtype=args.flowtype)
        validate_dataset = BaseFlow(aug_params=aug_params, split='validate', flowtype=args.validation[0])

    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size,
                                   pin_memory=False, shuffle=True, num_workers=2, drop_last=True)
    validate_dataset = data.DataLoader(validate_dataset, batch_size=args.batch_size,
                                       pin_memory=False, shuffle=True, num_workers=2, drop_last=True)

    print('Training with %d image pairs' % len(train_dataset))
    return train_loader, validate_dataset
