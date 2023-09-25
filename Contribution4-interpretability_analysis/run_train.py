import torch
import torch.nn as nn
import argparse

from data_loader import *
from training import train
from torch.optim.lr_scheduler import StepLR

from model import *
import wandb
# python main.py --batch_size 1 --source_dev_images ./Data-split/source_train_set_neg.csv --source_dev_masks ./Data-split/source_train_set_masks_neg.csv  --target_dev_images ./Data-split/target_train_set.csv --verbose True
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='initial learning rate')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size,  number of images in each iteration during training')
    parser.add_argument('--epochs', type=int, default=100, help='total epochs')
    parser.add_argument('--results_dir', type=str, default ="./results/", help='results directory')
    parser.add_argument('--source_csv', type=str, help='path to source dataset(images and labels)')
    parser.add_argument('--verbose', type=bool, default=False, help='verbose debugging flag')
    
    args = parser.parse_args()

    root_dir = args.results_dir # Path to store results
    verbose = args.verbose # Debugging flag
    
    # Set our data loaders - supervised training with no domain adaptation
    source_ds_train, source_train_loader, \
    source_ds_val, source_val_loader = load_data(args.source_csv, root_dir, batch = args.batch_size, verbose = verbose)
    
    # source_train_labels = np.array([target for (data, target, mask) in source_ds_train])
    source_train_labels = np.array([item['age_label'] for item in source_ds_train])
    # Check the shape and data type of the target labels
    print("Source train labels shape:", source_train_labels.shape)
    print("Source train labels data type:", source_train_labels.dtype)

    model = build_seq_sfcn()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5, betas=(0.5, 0.999))
    
    model = model.cuda()
    
    step = 20
    gamma = 0.5
    # Learning rate decay scheduler
    scheduler = StepLR(optimizer, step_size=step, gamma=gamma)


    run = wandb.init(
        project = "global-age-prediction",
        config = {
            "regressor": "SFCN",
            "learning_rate_initial": args.learning_rate,
            "batch": args.batch_size,
            "epochs": args.epochs,
            "scheduler_step": step,
            "scheduler_gamma": gamma,
            "loss": "mse"
        },
        name = "camcan-run4-mse-e100",
        job_type = "train"
    )

    print("Start of training...", flush=True)
    train(source_train_loader, source_val_loader,\
                model, optimizer, scheduler, args.epochs, root_dir, run)
    
    print("End of training.", flush=True)
        