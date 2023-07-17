import monai
from monai.handlers.utils import from_engine
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.metrics import DiceMetric
from monai.losses import DiceLoss, DiceCELoss
from monai.inferers import sliding_window_inference, SimpleInferer
from monai.data import CacheDataset, decollate_batch
from monai.config import print_config
from monai.apps import download_and_extract
import glob
import numpy as np
from monai.utils import first, set_determinism
from monai.data import Dataset, ArrayDataset, DataLoader, load_decathlon_datalist
from monai.transforms import (Transform,AsDiscrete,Activations, Activationsd, AddChanneld, Compose, LoadImaged,
                              Transposed, ScaleIntensityd, RandAxisFlipd, RandRotated, RandAxisFlipd,
                              RandBiasFieldd, ScaleIntensityRangePercentilesd, RandAdjustContrastd,
                              RandHistogramShiftd, DivisiblePadd, Orientationd, RandGibbsNoised, Spacingd,
                              RandRicianNoised, AsChannelLastd, RandSpatialCropd,ToNumpyd,EnsureChannelFirstd,
                              RandSpatialCropSamplesd, RandCropByPosNegLabeld)
from monai.config import print_config
from monai.metrics import DiceMetric
from monai.networks.nets import UNet, BasicUNet
from monai.data.utils import pad_list_data_collate
import pandas as pd
import random as rd


train_transforms = Compose(
    [
        LoadImaged(keys=["img", "seg_label", "brain_mask"]),
        # AddChanneld("img", "label"),
        EnsureChannelFirstd(keys=["img", "seg_label", "brain_mask"]),
        DivisiblePadd(["img", "seg_label", "brain_mask"], 16),
        # Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityd(
            keys=["img"],
            minv=0.0,
            maxv=1.0
        ),
        RandRotated(
            keys=["img", "seg_label", "brain_mask"],
            range_x=np.pi / 12,
            prob=0.5,
            keep_size=True,
            mode="nearest"
        ),
        RandCropByPosNegLabeld(
            keys=["img","seg_label", "brain_mask"],
            spatial_size=(128,128,128),
            label_key="brain_mask",
            pos = 0.7,
            neg=0.3,
            num_samples=1,
            image_key="img",
            image_threshold=-0.1
        )
    ]
)
val_transforms = Compose(
    [
        LoadImaged(keys=["img", "seg_label", "brain_mask"]),
        # AddChanneld("img", "label"),
        EnsureChannelFirstd(keys=["img", "seg_label", "brain_mask"]),
        DivisiblePadd(["img", "seg_label", "brain_mask"], 16),
        # Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityd(
            keys=["img"],
            minv=0.0,
            maxv=1.0
        ),
        RandRotated(
            keys=["img", "seg_label", "brain_mask"],
            range_x=np.pi / 12,
            prob=0.5,
            keep_size=True,
            mode="nearest"
        ),
        RandCropByPosNegLabeld(
            keys=["img","seg_label", "brain_mask"],
            spatial_size=(128,128,128),
            label_key="brain_mask",
            pos = 0.7,
            neg=0.3,
            num_samples=1,
            image_key="img",
            image_threshold=-0.1
        )
    ]
)


def load_data(t1w_csv, seg_mask_csv, age_csv_path, brain_mask_csv, batch, root_dir):
    #TODO: Add correct path to the csv with input file names and paths
    shuff_data = pd.read_csv(root_dir + "shuff_files_camcan.csv")
    imgs_list = list(shuff_data['imgs'])
    seg_labels = list(shuff_data['seg'])
    mask_dir = list(shuff_data['mask'])
    age_labels = list(shuff_data['age'])
    

    length = len(imgs_list)
    print(length)
    test = int(0.85*length)

    imgs_list = imgs_list[:test]
    seg_labels = seg_labels[:test]
    mask_dir = mask_dir[:test]
    age_labels = age_labels[:test]

    first = int(0.75*length)

    imgs_list_train = imgs_list[0:first]
    imgs_list_val = imgs_list[first:]
    seg_labels_train = seg_labels[0:first]
    seg_labels_val = seg_labels[first:]
    mask_dir_train = mask_dir[0:first]
    mask_dir_val = mask_dir[first:]
    age_labels_train = age_labels[0:first]
    age_labels_val = age_labels[first:]

    print('train set', len(imgs_list_train), len(seg_labels_train), len(mask_dir_train), len(age_labels_train))
    print('val set', len(imgs_list_val), len(seg_labels_val), len(mask_dir_val), len(age_labels_val))

    filenames_train = [{"img": x, "seg_label": y, "age_label": z, 'brain_mask': b} for (x,y,z,b) in zip(imgs_list_train, seg_labels_train, age_labels_train, mask_dir_train)]


    ds_train = monai.data.Dataset(filenames_train, train_transforms)
    print('ds train type', type(ds_train))
    train_loader = DataLoader(ds_train, batch_size=batch, shuffle = True, num_workers=2, pin_memory=True, collate_fn=pad_list_data_collate)

    filenames_val = [{"img": x, "seg_label": y, "age_label": z, 'brain_mask': b} for (x, y, z, b) in zip(imgs_list_val, seg_labels_val, age_labels_val, mask_dir_val)]

    ds_val = monai.data.Dataset(filenames_val, val_transforms)
    print('ds val type', type(ds_val))
    val_loader = DataLoader(ds_val, batch_size=batch, shuffle=True, num_workers=1, pin_memory=True, collate_fn=pad_list_data_collate)
    
    return ds_train, train_loader, ds_val, val_loader
