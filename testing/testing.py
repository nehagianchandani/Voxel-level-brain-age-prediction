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
from monai.inferers import sliding_window_inference, SimpleInferer, SlidingWindowInferer
from utils import *
from loss import *
import torch.nn.functional as F

def create_pad_mask(pred, target, brain_mask):
    for i in range(len(target.cpu().numpy())):

            ground_truth = torch.full_like(pred[i], target.cpu().numpy()[i]) #creates a tensor of size pred_shape and fills it with values target
            #noise = torch.randint_like(ground_truth, low=-2, high=3)
            ground_truth = torch.mul(ground_truth, brain_mask[i]) #tensor with target age values only in brain region

            prediction = torch.mul(pred[i], brain_mask[i]) #to ensure prediction tensor has non-zero values only in brain region
            loss_img = torch.sum(torch.abs(torch.sub(prediction,ground_truth, alpha=1))) / torch.sum(brain_mask[i]) #mean absolute error voxel wise
            pad_mask = torch.sub(prediction,ground_truth, alpha=1)
    return pad_mask, loss_img



def test_check1(test_loader, model, optimizer, scheduler, root_dir, path_to_chkpt):

    def model_fun1(input_img):
        pred_seg, pred_age, pred_vox_age = model(input_img)

        bg_ch = pred_seg[:, 0:1, :, :, :]
        f_ch = pred_seg[:, 1:2, :, :, :]
        s_ch = pred_seg[:, 2:3, :, :, :]
        t_ch = pred_seg[:, 3:4, :, :, :]

        pred_glob_age = torch.full_like(bg_ch, 0, dtype=torch.float32)

        for i in range(pred_age.size(0)):
            pred_glob_age[i,:,:,:,:] = pred_age[i].item()

        return bg_ch , f_ch, s_ch, t_ch, pred_glob_age, pred_vox_age

    state = torch.load(path_to_chkpt)
    model.load_state_dict(state['state_dict'])

    dir_name = str((path_to_chkpt.split('/')[-1]).split('.')[0])
    dir_name = 'camcan_1_' +(dir_name)
    path = os.path.join(root_dir,  dir_name)
    print(path)
    isExist = os.path.exists(path)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(path)
        print('dir created')

    with torch.no_grad():
        mae_loss = 0.0
        voxel_error = []
        metric_dice = []
        for step, batch in enumerate(test_loader):
            if step <50:
                img, age = (batch["img"].cuda(), batch["age_label"].cuda())
                print('--------------------------------------')
                print(batch["img"].meta["filename_or_obj"][0], age)
                # brain_img = img * brain_mask
                brain_img = img

                bg_ch , f_ch, s_ch, t_ch, pred_glob_age, pred_vox_age = sliding_window_inference(inputs=brain_img, roi_size=(128,128,128), sw_batch_size=8, predictor=model_fun1, overlap=0.9, progress=False)

                pred_seg = torch.cat((bg_ch, f_ch, s_ch, t_ch), dim=1)

                pred_seg = torch.argmax(pred_seg, dim=1)
                pred_seg = pred_seg.unsqueeze(0)
                print(pred_seg.shape, torch.unique(pred_seg), pred_glob_age.shape, pred_vox_age.shape)


                name = batch["img"].meta["filename_or_obj"][0].split('/')[-1]

                sub_dir = str(name)
                sub_path = os.path.join(path, sub_dir)
                isExist = os.path.exists(sub_path)
                if not isExist:
                    # Create a new directory because it does not exist
                    os.makedirs(sub_path)

                # save glob age image
                age_name = '/age_' + str(name)
                img1 = nib.load(batch["img"].meta["filename_or_obj"][0])
                age_output = pred_glob_age.squeeze(0).squeeze(0)
                age_output = nib.Nifti1Image(age_output.cpu().numpy(), img1.affine, img1.header)
                save = nib.save(age_output, str(sub_path) + str(age_name))

                # save seg image
                seg_name = '/seg_' + str(name)
                img1 = nib.load(batch["img"].meta["filename_or_obj"][0])
                seg_output = nib.Nifti1Image(pred_seg.cpu().numpy().squeeze(axis=0).squeeze(axis=0), img1.affine, img1.header)
                save = nib.save(seg_output, str(sub_path) + seg_name)


                # save voxel age image
                brain_mask = torch.where(brain_img > 0, 1, 0)
                brain_pad_mask, mae_voxel = create_pad_mask(pred_vox_age, age, brain_mask)
                print('voxel error avg for one img', mae_voxel.item())
                age_name = '/vox_age_pad_' + str(name)
                voxel_error.append(mae_voxel.item())
                img1 = nib.load(batch["img"].meta["filename_or_obj"][0])
                vox_age_output = brain_pad_mask.squeeze(0).squeeze(0)
                vox_age_output = nib.Nifti1Image(vox_age_output.cpu().as_tensor().cpu().numpy(), img1.affine, img1.header)
                save = nib.save(vox_age_output, str(sub_path) + age_name)

                # save orig image
                orig_name = '/orig_' + str(name)
                img1 = nib.load(batch["img"].meta["filename_or_obj"][0])
                orig = brain_img.squeeze(0).squeeze(0)
                orig_save = nib.Nifti1Image(orig.cpu().numpy(), img1.affine, img1.header)
                save = nib.save(orig_save, str(sub_path) + orig_name)
        print('--------------------------')
        print('avg voxel error on test set ', np.mean(voxel_error))











