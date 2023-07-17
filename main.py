import torch
import argparse
from model import *
from data_loader import *
from training import *
from verbose_utils import plot_samples_loaders
import wandb
import os


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='initial learning rate')
    parser.add_argument('--batch_size', type=int, default=2, help='batch size,  number of images in each iteration during training')
    parser.add_argument('--epochs', type=int, default=15, help='total epochs')
    parser.add_argument('--root_dir', type=str, help='root directory')
    parser.add_argument('--t1w_csv', type=str, help='path to t1w mri csv')
    parser.add_argument('--seg_mask_csv', type=str, help='path to segmentation mask csv')
    parser.add_argument('--age_csv_path', type=str, help='path to age info csv')
    parser.add_argument('--brain_mask_csv', type=str, help='path to binary brain mask csv')

    args = parser.parse_args()
    #add wandb project name
    wandb.init(project="xxxxx", settings=wandb.Settings(start_method="fork"))
    root_dir = args.root_dir


    verbose = False # debugging flag

    #add wandb run name
    wandb.run.name = 'xxxxxx'

    #Setting our data_loaders
    ds_train, train_loader, ds_val, val_loader = load_data(args.t1w_csv, args.seg_mask_csv, args.age_csv_path, args.brain_mask_csv, args.batch_size, root_dir)
    
    # Inspecting outputs of data loaders
    if verbose:
        plot_samples_loaders(train_loader, val_loader)

    # Building our 3D UNET model
    model = build_unet()
    model = model.cuda()
    
    # Defining our optimizer and scheduler
    # proposed model uses betas (0.5, 0.999)

    #ablation experiment models use default betas -- no need to define explicitly
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    
    # #after how many steps (in my case used for epochs) to change lr - step_size
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=70, gamma=0.6)

    max_epochs = args.epochs
    wandb.config = {
        "learning_rate": optimizer.param_groups[0]['lr'],
        "epochs": args.epochs,
        "batch_size": args.batch_size
    }
    print("Start of training...")
    train(train_loader, val_loader, model, optimizer, scheduler, max_epochs, root_dir)
    print("End of training...")
    
