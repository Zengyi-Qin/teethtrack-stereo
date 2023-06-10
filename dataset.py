import numpy as np
import json
from torch.utils.data import Dataset
import cv2
import os
from utils import convert_coco, affine_transform_and_crop, generate_keypoint_maps
from torchvision.transforms import RandAugment
import torch


class TeethImgAugment(RandAugment):
    def __init__(self):
        super().__init__()

    def _augmentation_space(self, num_bins, image_size):
        return {
            # op_name: (magnitudes, signed)
            "Identity": (torch.tensor(0.0), False),
            "Brightness": (torch.linspace(0.0, 0.9, num_bins), True),
            "Color": (torch.linspace(0.0, 0.9, num_bins), True),
            "Contrast": (torch.linspace(0.0, 0.9, num_bins), True),
            "Sharpness": (torch.linspace(0.0, 0.9, num_bins), True),
            "Posterize": (
                8 - (torch.arange(num_bins) / ((num_bins - 1) / 4)).round().int(),
                False,
            ),
            "AutoContrast": (torch.tensor(0.0), False),
            "Equalize": (torch.tensor(0.0), False),
        }


class TeethKptDataset(Dataset):
    def __init__(self, img_dir, anno_path, n_kpt=4, input_size=256, stride=4):
        self.img_dir = img_dir
        self.anno = json.load(open(anno_path))
        self.img_names = list(self.anno.keys())
        self.n_kpt = n_kpt
        self.input_size = input_size
        self.stride = stride
        self.aug = TeethImgAugment()

    def __len__(self):
        return len(self.anno)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img = cv2.imread(os.path.join(self.img_dir, img_name))
        kpts = np.array(self.anno[img_name]["keypoints"])
        conf = np.ones(kpts.shape[0])

        img, kpts = affine_transform_and_crop(
            img, kpts, (self.input_size, self.input_size)
        )
        hm = generate_keypoint_maps(img, kpts, conf, stride=self.stride, sigma=7)

        img_tensor = torch.tensor(np.transpose(img, (2, 0, 1)))
        img_tensor = self.aug(img_tensor)
        img_tensor = img_tensor / 256.0
        return img_tensor, hm
