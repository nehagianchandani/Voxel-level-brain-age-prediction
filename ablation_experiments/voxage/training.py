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
            glob_coef = 1
            vox_coef = 1
        elif (epoch >=50) and (epoch<100):
            glob_coef = 0.8
            vox_coef =1.2
        else:
            glob_coef = 0.6
            vox_coef =1.4

        for step, batch in enumerate(train_loader):
            img, brain_mask, tissue_mask, age = (batch["img"].cuda(), batch["brain_mask"].cuda(),
                                            batch["seg_label"].cuda(), batch["age_label"].cuda())

            brain_img = img*brain_mask
       
            optimizer.zero_grad()

            pred_tissue_mask, pred_glob_age, pred_voxel_age = model(brain_img)

            loss = glob_coef*(global_mae_loss(pred_glob_age, age, brain_mask)) + vox_coef*(voxel_mae(pred_vox_age, age, root_dir, brain_mask))
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            wandb.log({"lr": optimizer.param_groups[0]['lr']})
            print("=", end = "")

        train_loss = train_loss/(step+1)

        print()
        print("Val:", end ="")
        with torch.no_grad():
                glob_mae_loss=0.0
                vox_mae_loss = 0.0
                for step, batch in enumerate(val_loader):
                    img, brain_mask, tissue_mask, age = (batch["img"].cuda(), batch["brain_mask"].cuda(),
                                            batch["seg_label"].cuda(), batch["age_label"].cuda())
                    brain_img = img*brain_mask

                    pred_glob_age, pred_vox_age = model(brain_img)

                    loss = glob_coef * (global_mae_loss(pred_glob_age, age, brain_mask)) + vox_coef*(voxel_mae(pred_vox_age, age, root_dir, brain_mask))
                    val_loss += loss.item()

                    glob_mae_loss += global_mae_loss(pred_glob_age, age, brain_mask)
                    vox_mae_loss += voxel_mae(pred_vox_age, age, root_dir, brain_mask)

                    print("=", end = "")
                print()
                val_loss = val_loss/(step+1)
                glob_mae_loss = glob_mae_loss/ (step+1)
                vox_mae_loss = vox_mae_loss/ (step+1)


        print("Training epoch ", epoch, ", train loss:", train_loss, ", val loss:", val_loss, ", val glob mae:", glob_mae_loss.item(), ", val vox mae:", vox_mae_loss.item())
        wandb.log({"train_loss": train_loss})
        wandb.log({"val_loss": val_loss})
        wandb.log({"val_glob_mae": glob_mae_loss.item()})
        wandb.log({"val_vox_mae": vox_mae_loss.item()})
        if epoch == 1:
            best_val_loss = val_loss


        if val_loss < best_val_loss:
            print("Saving model")
            best_val_loss = val_loss
            state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            torch.save(state, root_dir + "/globage_voxage.pth")
        scheduler.step()
        wandb.log({"epoch": epoch})
    return
