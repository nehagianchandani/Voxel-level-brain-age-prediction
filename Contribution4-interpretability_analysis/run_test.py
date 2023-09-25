import torch
import torch.nn as nn

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

import monai
from monai.data import Dataset, DataLoader
from monai.transforms import (LoadImaged, EnsureChannelFirstd, ScaleIntensityd, \
                              RandAxisFlipd, RandGaussianNoised, RandGibbsNoised, \
                              RandSpatialCropd, Compose, DivisiblePadd, CropForegroundd, ResizeWithPadOrCropd, CenterSpatialCropd,NormalizeIntensityd, Orientationd)
from monai.metrics import MAEMetric, MSEMetric
from monai.visualize import GradCAM, SmoothGrad, OcclusionSensitivity, GuidedBackpropGrad, GuidedBackpropSmoothGrad
from monai.visualize.utils import blend_images
import nibabel as nib
import argparse
from data_loader import *
from model import *
import wandb

# import torchinfo
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default ="./results/", help='results directory')
    parser.add_argument('--source_csv', type=str, help='path to source test dataset(images and labels)')
    parser.add_argument('--model_path', type=str, help='path to trained model')
    parser.add_argument('--verbose', type=bool, default=False, help='verbose debugging flag')

    args = parser.parse_args()
    root_dir = args.results_dir
    verbose = args.verbose
    saved_model_path = args.model_path

    print('VERBOSE: ',verbose)
    
    run = wandb.init(
        project = "global-age-prediction",
        name = "camcan-run4-mse-cc359",
        job_type = "test"
    )

    source_test_ds, source_test_loader = load_test_data(args.source_csv, verbose=verbose)

    print('TEST SET: ', args.source_csv)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = build_seq_sfcn()
    
    model.cuda()

    model.load_state_dict(torch.load(saved_model_path)['state_dict'])

    model.eval()
    
    mse_object = MSEMetric()
    mae_object = MAEMetric()

    test_mae = 0
    test_mse = 0

    
    grad_cam_16 = GradCAM(nn_module=model, target_layers="features.12")
    grad_cam_20 = GradCAM(nn_module=model, target_layers="features.20")
    smooth_grad = SmoothGrad(model, n_samples=25, magnitude=True, stdev_spread=0.04)
    os = OcclusionSensitivity(nn_module=model, mask_size=20, n_batch=10, stride=20, activate=False)

    actual_ages = []
    predicted_ages = []
    test_mae_list = []

    columns = ["step", "chronological age", "predicted age", "mae"]
    predictions = wandb.Table(columns=columns)

    print("Start of testing...", flush=True)
    for step, batch in enumerate(source_test_loader):
        img, age, sid = (batch["img"].cuda(), batch["age_label"].cuda(), batch["sid"])
        sid = sid[0]
        age = age.unsqueeze(1)
        img.requires_grad = True

        output = model(img) 
        
        actual_age = age.item()
        predicted_age = output.item()

        wandb.log({
            "step": step,
            "sid":sid,
            "chronological age": actual_age,
            "predicted age": predicted_age
        })
        print(f"{sid} - Actual Age: {actual_age}, Predicted Age: {predicted_age}", flush=True)

        actual_ages.append(actual_age)
        predicted_ages.append(predicted_age)

        mae = mae_object(output, age)
        mse = mse_object(output.float(), age.float())
        
        predictions.add_data(step, actual_age, predicted_age, mae.item())

        test_mae += mae.item()
        test_mse += mse.item()

        test_mae_list.append(mae.item())

    
        gc_map_16 = grad_cam_16(x=img)
        gc_map_20 = grad_cam_20(x=img)

        gc_map_avg = torch.div(torch.add(gc_map_16, gc_map_20),2) 
        smoothg_map = smooth_grad(x=img)
        os_map, _ = os(x=img)  # Extract sensitivity map and ignore the mask

        sensitivity_map_numpy = os_map.cpu().numpy()
        vmin = sensitivity_map_numpy.min()
        vmax = sensitivity_map_numpy.max()
        bound = abs(predicted_age - vmin)
        if abs(vmax - predicted_age) > bound:
            bound = abs(vmax - predicted_age)

        
        #Plotting Axial views
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        # Plotting GradCam result
        axs[0].imshow(np.rot90(img.cpu().detach().numpy()[0,0,:,:,100], k=1), cmap='gray')
        im1 = axs[0].imshow(np.rot90(gc_map_avg.cpu().detach().numpy()[0, 0, :, :, 100], k=1), cmap='jet_r', alpha=0.7)
#        fig.colorbar(im1, ax=axs[0])
        axs[0].set_title('Grad-CAM')
            
        # Plotting SmoothGrad result
        axs[1].imshow(np.rot90(img.cpu().detach().numpy()[0,0,:,:,100], k=1), cmap='gray')
        im2 = axs[1].imshow(np.rot90(os_map.cpu().detach().numpy()[0, 0, :, :, 100], k=1), cmap='bwr', vmin = predicted_age-bound, vmax = predicted_age+bound, alpha=0.8)
#       fig.colorbar(im2, ax=axs[1])
        axs[1].set_title('Occlusion Sensitivity')

        
        axs[2].imshow(np.rot90(img.cpu().detach().numpy()[0,0,:,:,100], k=1), cmap='gray')
        im4 = axs[2].imshow(np.rot90(smoothg_map.cpu().detach().numpy()[0, 0, :, :, 100], k=1), cmap='jet', alpha=0.75)
#      fig.colorbar(im4, ax=axs[2])
        axs[2].set_title('SmoothGrad')

        # Removing axes for aesthetics
        for ax in axs:
            ax.axis('off')

        plt.tight_layout()
        plt.savefig(f"{root_dir}/{sid}_axial.png", dpi=500)  # Saving the figure



        #Plotting Sagittal views
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        # Plotting GradCam result
        axs[0].imshow(np.fliplr(np.rot90(img.cpu().detach().numpy()[0, 0, 100, :, :], k=1)), cmap='gray')
        im1 = axs[0].imshow(np.fliplr(np.rot90(gc_map_avg.cpu().detach().numpy()[0, 0, 100, :, :], k=1)), cmap='jet_r', alpha=0.7)
   #     fig.colorbar(im1, ax=axs[0])
        axs[0].set_title('Grad-CAM')

        # Plotting SmoothGrad result
        axs[1].imshow(np.fliplr(np.rot90(img.cpu().detach().numpy()[0, 0, 100, :, :], k=1)), cmap='gray')
        im2 = axs[1].imshow(np.fliplr(np.rot90(os_map.cpu().detach().numpy()[0, 0, 100, :, :], k=1)), cmap='bwr', vmin = predicted_age-bound, vmax = predicted_age+bound, alpha=0.8)
    #    fig.colorbar(im2, ax=axs[1])
        axs[1].set_title('Occlusion Sensitivity')

    
        axs[2].imshow(np.fliplr(np.rot90(img.cpu().detach().numpy()[0, 0, 100, :, :], k=1)), cmap='gray')
        im4 = axs[2].imshow(np.fliplr(np.rot90(smoothg_map.cpu().detach().numpy()[0, 0, 100, :, :], k=1)), cmap='jet', alpha=0.75)
     #   fig.colorbar(im4, ax=axs[2])
        axs[2].set_title('SmoothGrad')

        # Removing axes for aesthetics
        for ax in axs:
            ax.axis('off')

        plt.tight_layout()
        plt.savefig(f"{root_dir}/{sid}_sag.png", dpi=500)  # Saving the figure


        #Plotting Coronal views
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        # Plotting GradCam result
        axs[0].imshow(np.rot90(img.cpu().detach().numpy()[0, 0, :, 100, :], k=1), cmap='gray')
        im1 = axs[0].imshow(np.rot90(gc_map_avg.cpu().detach().numpy()[0, 0, :, 100, :], k=1), cmap='jet_r', alpha=0.7)
      #  fig.colorbar(im1, ax=axs[0])
        axs[0].set_title('Grad-CAM')

        # Plotting SmoothGrad result
        axs[1].imshow(np.rot90(img.cpu().detach().numpy()[0, 0, :, 100, :], k=1), cmap='gray')
        im2 = axs[1].imshow(np.rot90(os_map.cpu().detach().numpy()[0, 0, :, 100, :], k=1), cmap='bwr', vmin = predicted_age-bound, vmax = predicted_age+bound, alpha=0.8)
    #    fig.colorbar(im2, ax=axs[1])
        axs[1].set_title('Occlusion Sensitivity')

    
        axs[2].imshow(np.rot90(img.cpu().detach().numpy()[0, 0, :, 100, :], k=1), cmap='gray')
        im4 = axs[2].imshow(np.rot90(smoothg_map.cpu().detach().numpy()[0, 0, :, 100, :], k=1), cmap='jet', alpha=0.75)
        #fig.colorbar(im4, ax=axs[2])
        axs[2].set_title('SmoothGrad')

        # Removing axes for aesthetics
        for ax in axs:
            ax.axis('off')

        plt.tight_layout()
        plt.savefig(f"{root_dir}/{sid}_coronal.png", dpi=500)  # Saving the figure


    test_mae = test_mae/(step+1)
    test_mse = test_mse/(step+1)

    test_std = np.std(test_mae_list)
    

    print("test mae: ", test_mae, flush=True)
    print(f"test mae std: {test_std}", flush=True)

    print("test mse: ", test_mse, flush=True)

    columns = ["test_mae", "test_mae_std"]
    test_metrics = wandb.Table(columns=columns)
    test_metrics.add_data(test_mae, test_std)
    wandb.log({"test_metric_results": test_metrics})

    plt.figure(figsize=(10, 6))
    plt.scatter(actual_ages, predicted_ages, alpha=0.5)
    plt.xlabel("Actual Age")
    plt.ylabel("Predicted Age")

    # Plot a line where x = y
    plt.plot([min(actual_ages), max(actual_ages)], [min(actual_ages), max(actual_ages)], 'r')  

    plt.savefig(f"{root_dir}/actual_vs_predicted_age.png")  #Saving the figure
    print("End of testing")
    


    


