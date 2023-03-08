#multitask_1
import torch
from utils import *
from main import *

def dice_score(pred, target):
    eps = 0.0001
    dice = 0
    iflat = pred.reshape(-1)
    tflat = target.reshape(-1)
    intersection = (iflat * tflat).sum()
    union = iflat.sum() + tflat.sum()

    dice += (2.0 * intersection + eps) / (union + eps)
    if (intersection < 0 or union < 0 or dice > 1):
        print('min, max, sum of iflat predictions ', torch.min(iflat).item(), torch.max(iflat).item(),
              iflat.sum().item())
        print('min, max, sum of tflat labels ', torch.min(tflat).item(), torch.max(iflat).item(), iflat.sum().item())

    return dice

def mean_channel_dice(pred, target):
    dice = []
    for i in range(4):
        channel_val = dice_score(pred[:, i, :, :, :], target[:, i, :, :, :])
        # print('channel', i, 'dice score - ', channel_val,'\n')
        dice.append(channel_val)

    dice = torch.stack(dice, 0)
    # print('channel wise - ', dice.as_tensor())
    avg_dice= torch.mean(dice)
    #avg_dice = torch.find_mean(dice)

    return avg_dice

def find_mean(val):
    a, b, c, d = val.unbind(dim=0)
    foreground = ((1.1*b).add((1.2*c).add((1.2*d))))
    sum_val = (0.3*(a)).add(foreground)
    mean = torch.div(sum_val,4)

    return mean

def voxel_mae(pred, target, mask):

    brain_mask = mask
    #for testing
    #brain_mask = torch.randn((1, 1, 64, 64, 64)).cuda()
    #end for testing
    #print(pred.shape, target.shape)
    voxel_mae=[]
    zero_mask_ctr = 0
    #print('target and pred', target.shape, pred.shape)
    #print(target.cpu().numpy())
    for i in range(len(target.cpu().numpy())):
        #
        if torch.sum(brain_mask[i]) != 0:

            ground_truth = torch.full_like(pred[i], target.cpu().numpy()[i]) #creates a tensor of size pred_shape and fills it with values target
            noise = torch.randint_like(ground_truth, low=-2, high=3)
            noised_ground_truth = torch.add(ground_truth, noise)
            noised_ground_truth = torch.mul(noised_ground_truth, brain_mask[i]) #tensor with target age values only in brain region

            prediction = torch.mul(pred[i], brain_mask[i]) #to ensure prediction tensor has non-zero values only in brain region
            loss_img = torch.sum((torch.abs(prediction-noised_ground_truth))) / torch.sum(brain_mask[i]) #mean absolute error voxel wise
            voxel_mae.append(loss_img)
        else:
            zero_mask_ctr += 1
            ground_truth = torch.full_like(pred[i], target.cpu().numpy()[i])  # creates a tensor of size pred_shape and fills it with values target
            noise = torch.randint_like(ground_truth, low=-2, high=3)
            noised_ground_truth = torch.add(ground_truth, noise)
            dummy_brain_mask = torch.where(pred[i] > 0, 1, 0)  #to create a dummy brain mask to calc loss
            noised_ground_truth = torch.mul(noised_ground_truth, dummy_brain_mask)  # tensor with target age values only in brain region

            prediction = torch.mul(pred[i], dummy_brain_mask)  # to ensure prediction tensor has non-zero values only in brain region
            loss_img = torch.sum((torch.abs(prediction - noised_ground_truth))) / torch.sum(dummy_brain_mask)  # mean absolute error voxel wise
            # print('image loss', loss_img)
            if torch.any(loss_img.isnan()):
                # print('brain mask prediction gt sum ', torch.sum(dummy_brain_mask), torch.sum(prediction),torch.sum(ground_truth))
                loss_img = torch.nan_to_num(loss_img, 0)
            voxel_mae.append(loss_img)

        voxel_mae = torch.stack(voxel_mae, 0)
        # print(voxel_mae)
        loss = torch.mean(voxel_mae)
        #print('\nno of zero masks in this batch ', zero_mask_ctr)
        return loss

def global_mae_loss(pred, target):

    glob_mae = torch.abs(torch.sub(target, pred, alpha=1))
    glob_mae = torch.mean(glob_mae*1.0)
    return glob_mae

def overall_loss(seg_pred, seg_target, age_pred, age_target, root_dir, binary_mask, global_age, epoch):
    # print('calc overall loss')
    if epoch <= 50:
        mae_coef = 0
        dice_coef = 0
        globmae_coef = 1.0
    else:
        mae_coef = 0.7
        dice_coef = 0
        globmae_coef = 0.3

    mae = voxel_mae(age_pred, age_target, binary_mask)
    dice = mean_channel_dice(seg_pred, seg_target)
    global_mae = global_mae_loss(global_age, age_target)
    print('\nunweighted mae, global mae and dice score before loss calc ', mae, global_mae, dice)
    return (mae_coef * mae) + (dice_coef * ((-1) * dice)) + (globmae_coef * global_mae)

def overall_metrics(seg_pred, seg_target, age_pred, age_target, root_dir, binary_mask, global_age):
    #print('calc overall metric')
    return voxel_mae(age_pred, age_target, binary_mask), mean_channel_dice(seg_pred, seg_target), global_mae_loss(global_age, age_target)




