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
                              RandSpatialCropSamplesd, RandCropByPosNegLabeld,CenterSpatialCropd)
from monai.config import print_config
from monai.metrics import DiceMetric
from monai.networks.nets import UNet, BasicUNet
from monai.data.utils import pad_list_data_collate
import pandas as pd
import random as rd

train_transforms = Compose(
    [
        LoadImaged(keys=["img"]),
        # AddChanneld("img", "label"),
        EnsureChannelFirstd(keys=["img"]),
        #DivisiblePadd(["img", "brain_mask"], 16),
        # Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityd(
            keys=["img"],
            minv=0.0,
            maxv=1.0
        ),
        RandRotated(
            keys=["img"],
            range_x=np.pi / 12,
            prob=0.5,
            keep_size=True,
            mode="nearest"
        ),
    ]
)
val_transforms = Compose(
    [
        LoadImaged(keys=["img"]),
        # AddChanneld("img", "label"),
        EnsureChannelFirstd(keys=["img"]),
        #DivisiblePadd(["img", "brain_mask"], 16),
        # Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityd(
            keys=["img"],
            minv=0.0,
            maxv=1.0
        ),
        RandRotated(
            keys=["img"],
            range_x=np.pi / 12,
            prob=0.5,
            keep_size=True,
            mode="nearest"
        ),
    ]
)

test_transforms = Compose(
    [
        LoadImaged(keys=["img"]),
        # AddChanneld("img", "label"),
        EnsureChannelFirstd(keys=["img"]),
        # DivisiblePadd(["img", "brain_mask"], 64),   # use 64 instead of 16 when using non overlapping patches in sw inference
        # Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityd(
            keys=["img"],
            minv=0.0,
            maxv=1.0
        ),
    ]
)


def load_data(dataset_path, root_dir, batch, verbose=True):

    shuff_data = pd.read_csv(dataset_path)
    imgs_list = list(shuff_data['filename'])
    age_labels = list(shuff_data['age'])

    dev_test_split = int(0.85 * len(shuff_data)) #camcan dev test split

    imgs_list = imgs_list[:dev_test_split]
    age_labels = age_labels[:dev_test_split]
    
    length = len(imgs_list)
    print(length)

    # first = int(0.85 * length)
    first = int(0.75 * length) #camcan train val split

    imgs_list_train = imgs_list[0:first]
    imgs_list_val = imgs_list[first:]
    # mask_dir_train = mask_dir[0:first]
    # mask_dir_val = mask_dir[first:]
    age_labels_train = age_labels[0:first]
    age_labels_val = age_labels[first:]
    # if verbose: 
        # print('train set', len(imgs_list_train), len(mask_dir_train), len(age_labels_train), flush=True)
        # print('val set', len(imgs_list_val), len(mask_dir_val), len(age_labels_val), flush=True)

    # Creates dictionary to store nifti files' paths and their sex labels (1-> male; 0-> female)
    filenames_train = [{"img": x, "age_label": z} for (x,z) in zip(imgs_list_train,  age_labels_train)]

    # print('filenames train', filenames_train)
    ds_train = monai.data.Dataset(filenames_train, train_transforms)
    print('ds train type', type(ds_train), flush=True)
    train_loader = DataLoader(ds_train, batch_size=batch, shuffle = True, num_workers=0, pin_memory=True, collate_fn=pad_list_data_collate)

    filenames_val = [{"img": x, "age_label": z} for (x, z) in zip(imgs_list_val, age_labels_val)]

    ds_val = monai.data.Dataset(filenames_val, val_transforms)
    print('ds val type', type(ds_val), flush=True)
    val_loader = DataLoader(ds_val, batch_size=batch, shuffle=True, num_workers=0, pin_memory=True, collate_fn=pad_list_data_collate)
    
    return ds_train, train_loader, ds_val, val_loader

def load_test_data(dataset_path, verbose=False):
    shuff_data = pd.read_csv(dataset_path)
    imgs_list = list(shuff_data['filename'])
    age_labels = list(shuff_data['age'])
    subj_ids = list(shuff_data['subject_id'])
    print('# IMAGES: ', len(imgs_list))
    
    if verbose: #split test set from dataset
        dev_test_split = int(0.85 * len(shuff_data))
        imgs_list =  imgs_list[dev_test_split:]
        age_labels = age_labels[dev_test_split:]
        subj_ids = subj_ids[dev_test_split:]
    
    print(f"test set size (data loader): ", len(imgs_list))
    
    # verbose = False : test on another entire dataset
    filenames_test = [{"img": x, "age_label": z, "sid":s} for (x,z,s) in zip(imgs_list, age_labels, subj_ids)]

    # print('filenames train', filenames_train)
    ds_test = monai.data.Dataset(filenames_test, test_transforms)
    print('ds test type', type(ds_test), flush=True)
    test_loader = DataLoader(ds_test, batch_size=1, 
                                    shuffle=False, 
                                    num_workers=0, 
                                    pin_memory=True)

    return ds_test, test_loader