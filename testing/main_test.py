from monai.losses import DiceLoss
import torch
import matplotlib.pylab as plt
import numpy as np
from monai.metrics import DiceMetric
from monai.transforms import (
    AsDiscrete,
    Compose,
    EnsureType)
from monai.data.utils import pad_list_data_collate
from monai.data import decollate_batch
import wandb
from testing import *
import argparse
from model import *
from data_loader import *
import os

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='initial learning rate')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='batch size,  number of images in each iteration during training')
    parser.add_argument('--epochs', type=int, default=15, help='total epochs')
    parser.add_argument('--root_dir', type=str, help='root directory')
    parser.add_argument('--t1w_csv', type=str, help='path to t1w mri csv')
    parser.add_argument('--seg_mask_csv', type=str, help='path to segmentation mask csv')
    parser.add_argument('--age_csv_path', type=str, help='path to age info csv')
    parser.add_argument('--brain_mask_csv', type=str, help='path to binary brain mask csv')
    parser.add_argument('--checkpoint_path', type=str, help='path to checkpoint pth file')

    args = parser.parse_args()
    root_dir = args.root_dir


    verbose = False  # debugging flag

    ds_test, test_loader = load_data_test(args.t1w_csv, args.seg_mask_csv, args.age_csv_path,
                                                           args.brain_mask_csv, args.batch_size, root_dir)


    # Building our 3D UNET model
    model = build_unet()
    # print(model)
    model = model.cuda()
    # Defining our optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    # #after how many steps (in my case used for epochs, after 20 full epochs) to change lr - step_size
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=80, gamma=0.5)


    print("Start of training...")
    test_check1(test_loader, model, optimizer, scheduler, root_dir, args.checkpoint_path)
    print("End of training...")
