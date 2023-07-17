import matplotlib.pylab as plt
import numpy as np


# Lots of stuff hardcoded, which is bad programming practice,
# My goal was just to inspect the outputs from the data loaders
# to make sure they were correct
# I also looked into the intensities of the images.
def plot_samples_loaders(train_loader, val_loader, save_path = "./Figs/"):
    
    train_iter = iter(train_loader)
    val_iter = iter(val_loader)
    
    for ii in range(2):
    
        val_batch = next(val_iter)    
        train_batch = next(train_iter)
        print("train batch shape:", train_batch["img"].shape, 
                                    train_batch["brain_mask"].shape, 
                                    train_batch["seg_label"].shape)

        print("train batch min-max:", train_batch["img"].numpy().min(),
                                      train_batch["img"].numpy().max())
        
        print(np.unique(train_batch["brain_mask"].numpy()), 
              np.unique(train_batch["seg_label"].numpy()))
        
        print("val batch shape:", val_batch["img"].shape, val_batch["brain_mask"].shape, 
                                  val_batch["seg_label"].shape)
        
        plt.figure()
        plt.subplot(131)
        plt.imshow(train_batch["img"][0,0,32,:,:], cmap = "gray")
        plt.subplot(132)
        plt.imshow(train_batch["img"][0,0,32,:,:], cmap = "gray")
        plt.imshow(train_batch["brain_mask"][0,0,32,:,:], alpha = 0.4)
        plt.subplot(133)
        plt.imshow(train_batch["img"][0,0,32,:,:], cmap = "gray")
        plt.imshow(train_batch["seg_label"][0,0,32,:,:], alpha = 0.4)
        plt.savefig("./Figs/train_sample" + str(ii) + ".png")

        plt.figure()
        plt.subplot(131)
        plt.imshow(val_batch["img"][0,0,32,:,:], cmap = "gray")
        plt.subplot(132)
        plt.imshow(val_batch["img"][0,0,32,:,:], cmap = "gray")
        plt.imshow(val_batch["brain_mask"][0,0,32,:,:], alpha = 0.4)
        plt.subplot(133)
        plt.imshow(val_batch["img"][0,0,32,:,:], cmap = "gray")
        plt.imshow(val_batch["seg_label"][0,0,32,:,:], alpha = 0.4)
        plt.savefig("./Figs/val_sample" + str(ii) + ".png")
    return