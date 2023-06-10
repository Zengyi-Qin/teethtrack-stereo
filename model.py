import torch
from torch import nn
import torch.nn.functional as F


class TeethKptNet(nn.Module):
    def __init__(self, n_kpt):
        super(TeethKptNet, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, 7, 1, 3)
        self.conv1_2 = nn.Conv2d(64, 64, 3, 1, 1)

        self.conv2_1 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, 1, 1)

        self.conv3_1 = nn.Conv2d(128, 256, 3, 1, 1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, 1, 1)

        self.conv4_1 = nn.Conv2d(256, 256, 3, 1, 2, 2)
        self.conv4_2 = nn.Conv2d(256, 256, 3, 1, 2, 2)
        self.conv4_3 = nn.Conv2d(256, 256, 3, 1, 2, 2)

        self.conv5_1 = nn.Conv2d(512, 256, 3, 1, 1)
        self.conv5_2 = nn.Conv2d(256, 256, 3, 1, 1)

        self.conv_out = nn.Conv2d(256, n_kpt, 1, 1)

        self.relu = nn.ReLU()
        self.max_pooling = nn.MaxPool2d(3, 2, 1)

    def forward(self, x):
        x1_1 = self.relu(self.conv1_1(x))
        x1_2 = self.relu(self.conv1_2(x1_1))

        x2_1 = self.relu(self.conv2_1(self.max_pooling(x1_2)))
        x2_2 = self.relu(self.conv2_2(x2_1))

        x3_1 = self.relu(self.conv3_1(self.max_pooling(x2_2)))
        x3_2 = self.relu(self.conv3_2(x3_1))

        x4_1 = self.relu(self.conv4_1(self.max_pooling(x3_2)))
        x4_2 = self.relu(self.conv4_2(x4_1))
        x4_3 = self.relu(self.conv4_3(x4_2))

        x5_1 = self.relu(
            self.conv5_1(
                torch.cat(
                    [F.interpolate(x4_3, scale_factor=2, mode="bilinear"), x3_2], dim=1
                )
            )
        )
        x5_2 = self.relu(self.conv5_2(x5_1))

        hm_pred = self.conv_out(x5_2)

        return hm_pred
