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
from loss import global_mae_loss
from loss import voxel_mae


def train(train_loader, val_loader, model, optimizer, scheduler, max_epochs, root_dir):
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.train()
    
    best_val_loss = 0.0
    best_dice_metric = 0.0
    best_mae_score = 0.0
    best_voxel_mae_score = 0.0
    post_seglabel = Compose([EnsureType("tensor"), AsDiscrete(to_onehot=4)])
    loss_object = DiceLoss(to_onehot_y = True)
    metric_object = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    for epoch in range(1,max_epochs +1):
        train_loss = 0.0
        val_loss = 0.0
    
        print("Epoch ", epoch)
        print("Train:", end ="")
        if epoch < 50:
            dice_coef = 80
            glob_coef = 1
            voxel_coef = 1

        elif epoch >= 50 and epoch < 130:
            dice_coef = 40
            glob_coef = 1
            voxel_coef = 1

        else:
            dice_coef = 15
            glob_coef = 0.7
            voxel_coef = 1.3


        for step, batch in enumerate(train_loader):
            img, brain_mask, tissue_mask, age = (batch["img"].cuda(), batch["brain_mask"].cuda(),
                                            batch["seg_label"].cuda(), batch["age_label"].cuda())

            brain_img = img*brain_mask
       
            optimizer.zero_grad()

            pred_tissue_mask, pred_glob_age, pred_voxel_age = model(brain_img)

            loss = (dice_coef*loss_object(pred_tissue_mask,tissue_mask)) + (glob_coef*global_mae_loss(pred_glob_age, age, brain_mask)) + (voxel_coef*voxel_mae(pred_voxel_age, age, brain_mask))
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            wandb.log({"lr": optimizer.param_groups[0]['lr']})
            print("=", end = "")

        train_loss = train_loss/(step+1)

        print()
        print("Val:", end ="")
        with torch.no_grad():
                mae_loss=0.0
                voxel_mae_loss=0.0
                for step, batch in enumerate(val_loader):
                    img, brain_mask, tissue_mask, age = (batch["img"].cuda(), batch["brain_mask"].cuda(),
                                            batch["seg_label"].cuda(), batch["age_label"].cuda())
                    brain_img = img*brain_mask

                    pred_tissue_mask, pred_glob_age, pred_voxel_age = model(brain_img)


                    loss = (dice_coef*loss_object(pred_tissue_mask,tissue_mask)) + (glob_coef*global_mae_loss(pred_glob_age, age, brain_mask)) + (voxel_coef*voxel_mae(pred_voxel_age, age, brain_mask))
                    val_loss += loss.item()
                    tissue_mask = [post_seglabel(i) for i in decollate_batch(tissue_mask)]
                    tissue_mask = pad_list_data_collate(tissue_mask)
                    metric_object(y_pred=pred_tissue_mask, y=tissue_mask)
                    mae_loss += global_mae_loss(pred_glob_age, age, brain_mask)
                    voxel_mae_loss += voxel_mae(pred_voxel_age, age, brain_mask)

                    print("=", end = "")
                print()
                val_loss = val_loss/(step+1)
                dice_metric = metric_object.aggregate().item()
                metric_object.reset()
                mae_loss = mae_loss/ (step+1)
                voxel_mae_loss = voxel_mae_loss / (step + 1)


        print("Training epoch ", epoch, ", train loss:", train_loss, ", val loss:", val_loss, ", val dice score:", dice_metric, ", val glob mae:", mae_loss.item(), ", val voxel mae:", voxel_mae_loss.item(), " | ", optimizer.param_groups[0]['lr'])
        wandb.log({"train_loss": train_loss})
        wandb.log({"val_loss": val_loss})
        wandb.log({"val_dice_metric": dice_metric})
        wandb.log({"mae_global": mae_loss.item()})
        wandb.log({"mae_voxel": voxel_mae_loss.item()})
        if epoch == 1:
            best_val_loss = val_loss
            best_val_dice = dice_metric
            best_mae_score = mae_loss.item()
            best_voxel_mae_score = voxel_mae_loss.item()
        if val_loss < best_val_loss:
            print("Saving model")
            best_val_loss = val_loss
            best_dice_metric = dice_metric
            best_mae_score = mae_loss.item()
            best_voxel_mae_score = voxel_mae_loss.item()
            state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }

            torch.save(state, root_dir + "/saved_model.pth")
        scheduler.step()
        wandb.log({"epoch": epoch})
    return
