import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
from dataset import TeethKptDataset
from model import TeethKptNet
from utils import draw_keypoints

import argparse
from tqdm import tqdm


def vis(imgs, hms_pred, epoch, vis_dir, n_vis=10):
    for idx in range(min(n_vis, imgs.shape[0])):
        img = imgs[idx].detach().cpu().numpy()
        img = np.clip(np.transpose(img * 256, (1, 2, 0)), 0, 255).astype(np.uint8)
        img = np.ascontiguousarray(img)
        hm = hms_pred[idx].detach().cpu().numpy()
        img_vis = draw_keypoints(img, hm)
        img_vis = img
        save_path = os.path.join(
            vis_dir, f"epoch_{str(epoch).zfill(3)}_{str(idx).zfill(3)}.png"
        )
        cv2.imwrite(save_path, img_vis)


def train(model, dataloader, optimizer, device, epoch, vis_dir):
    model.train()
    loss_mean = []

    for imgs, hms in tqdm(dataloader):
        imgs = imgs.to(device)
        hms = hms.to(device)

        hms_pred = model(imgs)
        loss = torch.mean((hms_pred - hms) ** 2) * 1000

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_mean.append(loss.detach().cpu().numpy())

    vis(imgs, hms_pred, epoch, vis_dir)
    print(
        "Epoch {}, LR: {}, Loss: {:.3f}".format(
            epoch, optimizer.param_groups[0]["lr"], np.mean(loss_mean)
        )
    )


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TeethKptNet(n_kpt=4)
    model.to(device)
    model = torch.nn.DataParallel(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = MultiStepLR(
        optimizer,
        milestones=[int(args.epochs * 0.7), int(args.epochs * 0.9)],
        gamma=0.1,
    )

    save_dir = args.ckp
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    vis_dir = args.vis
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)

    train_dataset = TeethKptDataset(
        os.path.join(args.data, "images"), os.path.join(args.data, "anno.json")
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batchsize, shuffle=True, num_workers=args.workers
    )

    for i in range(args.epochs):
        train(model, train_loader, optimizer, device, i, vis_dir)
        scheduler.step()
        torch.save(
            model.state_dict(),
            os.path.join(save_dir, "checkpoint_{}.pth".format(str(i).zfill(3))),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batchsize", type=int, default=16)
    parser.add_argument(
        "--workers", type=int, default=32, help="number of workers in dataloader"
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="number of epochs in training"
    )
    parser.add_argument("--data", default="./data/train", help="dataset root")
    parser.add_argument(
        "--ckp", default="./outputs/checkpoint", help="path to save checkpoints"
    )
    parser.add_argument(
        "--vis", default="./outputs/vis", help="path to save visualizations"
    )

    args = parser.parse_args()

    main(args)
