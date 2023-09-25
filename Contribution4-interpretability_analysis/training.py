from monai.losses import DiceLoss
import torch
import torch.nn as nn
import matplotlib.pylab as plt
import numpy as np
from monai.metrics import DiceMetric, MAEMetric
from monai.transforms import (
    AsDiscrete,
    Compose,
    EnsureType)
from monai.data.utils import pad_list_data_collate
from monai.data import decollate_batch
import wandb

def train(train_loader, val_loader, model, optimizer, scheduler, max_epochs, root_dir, run):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.train()

    best_val_loss = 0.0
    # best_mae_score = 0.0

    # loss_object = nn.L1Loss()
    loss_object = nn.MSELoss()

    for epoch in range(1, max_epochs + 1):
        train_loss = 0.0
        val_loss = 0.0

        print("Epoch ", epoch, flush=True)
        print("Train:", end="", flush=True)

        for step, batch in enumerate(train_loader):
            img, age = (batch["img"].cuda(), batch["age_label"].cuda())
            
            age = age.unsqueeze(1)
            # brain_img = img*brain_mask
            brain_img = img

            optimizer.zero_grad()

            pred_glob_age = model(brain_img)

            loss = loss_object(pred_glob_age.float(), age.float())
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            print("=", end="", flush=True)

        train_loss = train_loss / (step + 1)

        print()
        print("Val:", end="", flush=True)
        with torch.no_grad():
            # mae_loss = 0.0

            for step, batch in enumerate(val_loader):
                img, age = (batch["img"].cuda(), batch["age_label"].cuda())

                age = age.unsqueeze(1)
                brain_img = img

                pred_glob_age = model(brain_img)

                loss = loss_object(pred_glob_age.float(), age.float())
                val_loss += loss.item()
                # mae_loss += loss_object(pred_glob_age.float(), age.float())

                print("=", end="", flush=True)
            print()
            val_loss = val_loss / (step + 1)
            # mae_loss = mae_loss / (step + 1)
            # print(mae_loss)

        print("Training epoch ", epoch, ", train loss:", train_loss, ", val loss:", val_loss, " | ", optimizer.param_groups[0]['lr'], flush=True)
        
        wandb.log({"epoch":epoch,
                   "train_loss": train_loss,
                   "val_loss": val_loss,
                #    "val global mae": mae_loss.item(),
                   "learning_rate": optimizer.param_groups[0]["lr"]})

        if epoch == 1:
            print("Saving model - first iter", flush=True)
            best_val_loss = val_loss
            # best_mae_score = mae_loss.item()

        if val_loss < best_val_loss:
            print("Saving model", flush=True)
            best_val_loss = val_loss
            # best_mae_score = mae_loss.item()
            wandb.run.summary["best_val_loss"] = best_val_loss
            # wandb.run.summary["best_mae_score"] = best_mae_score
            wandb.run.summary["best_model_epoch"] = epoch
            state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }

            torch.save(state, root_dir + "/sfcn_best_model.pth")
        scheduler.step()

    return
