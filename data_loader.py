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
        #RandSpatialCropSamplesd(
        #    keys=["img","seg_label", "brain_mask"],
        #    roi_size=(64,64,64),
        #    num_samples= 1,
        #    random_size=False
        #)
        RandCropByPosNegLabeld(
            keys=["img","seg_label", "brain_mask"],
            spatial_size=(96, 96, 96),
            label_key="brain_mask",
            pos = 0.7,
            neg=0.3,
            num_samples=1,
            image_key="img",
            image_threshold=-0.1
        )

        # RandGibbsNoised(keys=["img", "label"], prob = 0.2)
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
        #RandSpatialCropSamplesd(
        #    keys=["img", "seg_label", "brain_mask"],
        #    roi_size=(64, 64, 64),
        #    num_samples=1,
        #    random_size=False
        #)
        RandCropByPosNegLabeld(
            keys=["img","seg_label", "brain_mask"],
            spatial_size=(96, 96, 96),
            label_key="brain_mask",
            pos = 0.7,
            neg=0.3,
            num_samples=1,
            image_key="img",
            image_threshold=-0.1
        )

        # RandGibbsNoised(keys=["img", "label"], prob = 0.2)
    ]
)


def load_data(t1w_csv, seg_mask_csv, age_csv_path, brain_mask_csv, batch, root_dir):


    #t1w = pd.read_csv(t1w_csv)
    #tissue = pd.read_csv(seg_mask_csv)
    #brain = pd.read_csv(brain_mask_csv)
    #age = pd.read_csv(age_csv_path)
    #print(t1w.shape, tissue.shape, brain.shape, age.shape)
    #imgs_list = list(t1w['filename'])
    #seg_labels = list(tissue['filename'])
    #mask_dir = list(brain['filename'])
    #age_labels = list(age['age'])

    #imgs_list = imgs_list[1155:1514]
    #seg_labels = seg_labels[1155:1514]
    #mask_dir = mask_dir[1155:1514]
    #age_labels = age_labels[1155:1514]

    #fixed_seed = rd.random()
    #print('random shuffle seed', fixed_seed)
    #rd.Random(fixed_seed).shuffle(imgs_list)
    #rd.Random(fixed_seed).shuffle(seg_labels)
    #rd.Random(fixed_seed).shuffle(mask_dir)
    #rd.Random(fixed_seed).shuffle(age_labels)

    #shuff_dict = {'imgs': imgs_list, 'seg': seg_labels, 'mask': mask_dir, 'age': age_labels}
    #shuff_df = pd.DataFrame(shuff_dict)
    #shuff_df.to_csv(root_dir + "shuff_files.csv", encoding='utf-8', index=False)




    #add path to csv with paths of all data inputs
    shuff_data = pd.read_csv(root_dir + "shuff_files_1.csv")
    imgs_list = list(shuff_data['imgs'])
    seg_labels = list(shuff_data['seg'])
    mask_dir = list(shuff_data['mask'])
    age_labels = list(shuff_data['age'])

    imgs_list = imgs_list[0:329]
    seg_labels = seg_labels[0:329]
    mask_dir = mask_dir[0:329]
    age_labels = age_labels[0:329]


    length = len(imgs_list)
    print(length)
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
    #filenames_train = [{"img": x, "seg_label": y, 'brain_mask': b} for (x,y,b) in zip(imgs_list_train, seg_labels_train, mask_dir_train)]


    ds_train = monai.data.Dataset(filenames_train, train_transforms)
    print('ds train type', type(ds_train))
    train_loader = DataLoader(ds_train, batch_size=batch, shuffle = True, num_workers=2, pin_memory=True, collate_fn=pad_list_data_collate)

    filenames_val = [{"img": x, "seg_label": y, "age_label": z, 'brain_mask': b} for (x, y, z, b) in zip(imgs_list_val, seg_labels_val, age_labels_val, mask_dir_val)]

    ds_val = monai.data.Dataset(filenames_val, val_transforms)
    print('ds val type', type(ds_val))
    val_loader = DataLoader(ds_val, batch_size=batch, shuffle=True, num_workers=1, pin_memory=True, collate_fn=pad_list_data_collate)
    
    return ds_train, train_loader, ds_val, val_loader