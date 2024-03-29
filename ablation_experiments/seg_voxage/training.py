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
            dice_coef = 50
        elif epoch >=50 and epoch<100:
            dice_coef = 10
        else:
            dice_coef = 5

        for step, batch in enumerate(train_loader):
            img, brain_mask, tissue_mask, age = (batch["img"].cuda(), batch["brain_mask"].cuda(),
                                            batch["seg_label"].cuda(), batch["age_label"].cuda())

            brain_img = img*brain_mask
       
            optimizer.zero_grad()

            pred_tissue_mask, pred_vox_age = model(brain_img)

            loss = (dice_coef*loss_object(pred_tissue_mask,tissue_mask)) + voxel_mae(pred_vox_age, age, root_dir, brain_mask)
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
                for step, batch in enumerate(val_loader):
                    img, brain_mask, tissue_mask, age = (batch["img"].cuda(), batch["brain_mask"].cuda(),
                                            batch["seg_label"].cuda(), batch["age_label"].cuda())
                    brain_img = img*brain_mask

                    pred_tissue_mask, pred_vox_age = model(brain_img)


                    loss = (dice_coef*loss_object(pred_tissue_mask,tissue_mask)) + voxel_mae(pred_vox_age, age, root_dir, brain_mask)
                    val_loss += loss.item()
                    tissue_mask = [post_seglabel(i) for i in decollate_batch(tissue_mask)]
                    tissue_mask = pad_list_data_collate(tissue_mask)
                    metric_object(y_pred=pred_tissue_mask, y=tissue_mask)
                    mae_loss += voxel_mae(pred_vox_age, age, root_dir, brain_mask)

                    print("=", end = "")
                print()
                val_loss = val_loss/(step+1)
                dice_metric = metric_object.aggregate().item()
                metric_object.reset()
                mae_loss = mae_loss/ (step+1)
                #print(mae_loss)
        if epoch % 100 == 0:
                img = img.cpu()
                pred_tissue_mask = pred_tissue_mask.cpu()
                plt.figure()
                plt.subplot(121)
                plt.imshow(img.numpy()[0,0,32,:,:], cmap = "gray")
                plt.subplot(122)
                plt.imshow(img.numpy()[0,0,32,:,:], cmap = "gray")
                plt.imshow(np.argmax(pred_tissue_mask.numpy(),axis = 1)[0,32,:,:], alpha = 0.4)
                plt.savefig("./Figs1/val_sample_epoch_" + str(epoch) + ".png")

        print("Training epoch ", epoch, ", train loss:", train_loss, ", val loss:", val_loss, ", val dice score:", dice_metric, ", val mae score:", mae_loss.item())
        wandb.log({"train_loss": train_loss})
        wandb.log({"val_loss": val_loss})
        wandb.log({"val_dice_metric": dice_metric})
        wandb.log({"mae_voxel": mae_loss.item()})
        if epoch == 1:
            best_val_loss = val_loss
            best_val_dice = dice_metric
            best_mae_score = mae_loss.item()
        if val_loss < best_val_loss:
            print("Saving model")
            best_val_loss = val_loss
            best_dice_metric = dice_metric
            best_mae_score = mae_loss.item()
            state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            #torch.save(model.state_dict(), root_dir + "/seg_age_network.pth")
            torch.save(state, root_dir + "/seg_voxage_1.pth")
        scheduler.step()
        wandb.log({"epoch": epoch})
    return
