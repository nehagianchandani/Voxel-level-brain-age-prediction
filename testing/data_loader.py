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
import os

test_transforms = Compose(
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
        )


    ]
)


def load_data_test(t1w_csv, seg_mask_csv, age_csv_path, brain_mask_csv, batch, root_dir):

    file_name = "shuff_files_1.csv"
    files = os.path.join(root_dir, file_name)
    shuff_data = pd.read_csv(files)
    imgs_list = list(shuff_data['imgs'])
    seg_labels = list(shuff_data['seg'])
    mask_dir = list(shuff_data['mask'])
    age_labels = list(shuff_data['age'])
    imgs_list = imgs_list[329:]
    seg_labels = seg_labels[329:]
    mask_dir = mask_dir[329:]
    age_labels = age_labels[329:]


    filenames_test = [{"img": x, "seg_label": y, "age_label": z, 'brain_mask': b} for (x, y, z, b) in
                      zip(imgs_list, seg_labels, age_labels, mask_dir)]


    # print('filenames train', filenames_train)
    ds_test = monai.data.Dataset(filenames_test, test_transforms)
    test_loader = DataLoader(ds_test, batch_size=1, shuffle=True, num_workers=2, pin_memory=True,
                             collate_fn=pad_list_data_collate)

    return ds_test, test_loader
