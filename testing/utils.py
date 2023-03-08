import torch
from torch.optim.lr_scheduler import StepLR
import pytorch_lightning
from monai.utils import set_determinism
import monai
from monai.transforms import (
    AsDiscrete,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    ScaleIntensityRanged,
    Spacingd,
    EnsureType,
    DivisiblePadd,
    ScaleIntensityd,
    RandRotated
)
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.metrics import DiceMetric
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference, SimpleInferer
from monai.data import CacheDataset, Dataset, list_data_collate, decollate_batch, DataLoader
from monai.data.utils import pad_list_data_collate
from monai.config import print_config
from monai.apps import download_and_extract
import matplotlib.pyplot as plt
import tempfile
import shutil
import os
import glob
import numpy as np
from tqdm.notebook import tqdm_notebook
import torch.nn.functional as Fun
import nibabel as nib
from main import *



post_segpred = Compose([EnsureType("tensor", device="cpu")])
post_seglabel = Compose([EnsureType("tensor", device="cpu"), AsDiscrete(to_onehot=4)])
#post_agepred = Compose([EnsureType("tensor", device="cpu")])
post_agelabel = Compose([EnsureType("tensor", device="cpu")])

post_predseg = Compose([EnsureType("tensor", device="cpu"), AsDiscrete(to_onehot=4)])


step_loss_values=[]
dicemetric_values=[]
maemetric_values=[]
globmaemetric_values =[]
dice_val_best = 0.0
mae_val_best = 0.0
global_step = 0
global_step_best = 0

eval_num = 1