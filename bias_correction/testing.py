import glob
import numpy as np
import nibabel as nib
import pandas as pd
import torch
import matplotlib.pyplot as plt
import os



def test_before_correct():

    #create csv with test file paths and age info and read it in the next line
    testing = pd.read_csv('xxx')

    files = list(testing['imgs'])
    age = list(testing['age'])
    
    #print(files, age)
    print(len(files), len(age))
    vox_preds = []
    total = 0
    
    for file, a in zip(files, age):
        img = nib.load(file)
        img_data = img.get_fdata()

        avg_error = np.sum((np.abs(img_data))) / np.count_nonzero(img_data)
        str_avg = str(avg_error)
        #add path to text file to save results
        with open('xxxxxx', 'a') as f:
            f.write(str_avg)
            f.write('\n')
        print('mae error {:.4f}'.format(avg_error))
        total += avg_error

        # plotting
        pad = torch.from_numpy(img_data)
        mask = torch.where(pad != 0, 1, 0)
        chr_tensor = torch.full_like(pad, a, dtype=torch.float32)
        chr_age = chr_tensor * mask
        # print(chr_age.shape, torch.unique(chr_age), chr_age)
        vox_pred = torch.add(pad, chr_age)
        # print(vox_pred.shape, torch.unique(vox_pred), vox_pred)
        vox_pred = vox_pred * mask
        # print(vox_pred.shape, torch.unique(vox_pred), vox_pred)
        vox_error = torch.sum(vox_pred) / torch.sum(mask)
        vox_preds.append(vox_error.item())

    plt.figure()
    # .scatter(chron_age, vox_preds)
    plt.xlim([(min(age) - 2), (max(age) + 2)])
    plt.ylim([(min(vox_preds) - 2), (max(vox_preds) + 2)])
    plt.xlabel('chronological age')
    plt.ylabel('pred age')
    plt.axline((0, 0), slope=1)
    for i, (vox, chr) in enumerate(zip(vox_preds, age)):
        plt.plot(chr, vox, 'o', color='red')
    #add path to save plot
    plt.savefig('xxxxxx')

    print('test set avg', total/len(files))

    test_results = total/len(files)
    str_test = str(test_results)
    #add path to same text file as before
    with open('xxxxxx', 'a') as f:
        f.write('\n***test set avg****')
        f.write(str_test)

    

def test_after_correct():
    # create csv with test file paths and age info and read it in the next line
    testing = pd.read_csv('xxx')

    files = list(testing['imgs'])
    age = list(testing['age'])


    vox_preds =[]
    total = 0
    for i, a in zip(files, age):
        if a < 45:
            bin = nib.load(r"xxxxx path to bin 1")
            bin_data = bin.get_fdata()
            img = nib.load(i)
            img_data = img.get_fdata()

            corrected_pad = np.subtract(img_data, bin_data)
            name = i.split('_transformed.nii.gz')[0]
            corrected_pad_name=str(name)+'_corrected_pad.nii.gz'
            corrected_pad = np.subtract(img_data, bin_data)
            corrected_img=nib.Nifti1Image(corrected_pad,img.affine, img.header)
            print(corrected_pad_name)
            nib.save(corrected_img, corrected_pad_name)

        elif a >= 45 and a < 50:
            bin = nib.load(r"xxxxx path to bin 2")
            bin_data = bin.get_fdata()
            img = nib.load(i)
            img_data = img.get_fdata()

            corrected_pad = np.subtract(img_data, bin_data)
            name = i.split('_transformed.nii.gz')[0]
            corrected_pad_name=str(name)+'_corrected_pad.nii.gz'
            corrected_pad = np.subtract(img_data, bin_data)
            corrected_img=nib.Nifti1Image(corrected_pad,img.affine, img.header)
            print(corrected_pad_name)
            nib.save(corrected_img, corrected_pad_name)

        elif a >=50 and a < 55:
            bin = nib.load(r"xxxxx path to bin 3")
            bin_data = bin.get_fdata()
            img = nib.load(i)
            img_data = img.get_fdata()

            corrected_pad = np.subtract(img_data, bin_data)
            name = i.split('_transformed.nii.gz')[0]
            corrected_pad_name=str(name)+'_corrected_pad.nii.gz'
            corrected_pad = np.subtract(img_data, bin_data)
            corrected_img=nib.Nifti1Image(corrected_pad,img.affine, img.header)
            nib.save(corrected_img, corrected_pad_name)

        elif a >= 55 and a < 60:
            bin = nib.load(r"xxxxx path to bin 4")
            bin_data = bin.get_fdata()
            img = nib.load(i)
            img_data = img.get_fdata()

            corrected_pad = np.subtract(img_data, bin_data)
            name = i.split('_transformed.nii.gz')[0]
            corrected_pad_name=str(name)+'_corrected_pad.nii.gz'
            corrected_pad = np.subtract(img_data, bin_data)
            corrected_img=nib.Nifti1Image(corrected_pad,img.affine, img.header)
            nib.save(corrected_img, corrected_pad_name)

        elif a >= 60 and a < 65:
            bin = nib.load(r"xxxxx path to bin 5")
            bin_data = bin.get_fdata()
            img = nib.load(i)
            img_data = img.get_fdata()

            corrected_pad = np.subtract(img_data, bin_data)
            name = i.split('_transformed.nii.gz')[0]
            corrected_pad_name=str(name)+'_corrected_pad.nii.gz'
            corrected_pad = np.subtract(img_data, bin_data)
            corrected_img=nib.Nifti1Image(corrected_pad,img.affine, img.header)
            nib.save(corrected_img, corrected_pad_name)

        elif a >= 65:
            bin = nib.load(r"xxxxx path to bin 6")
            bin_data = bin.get_fdata()
            img = nib.load(i)
            img_data = img.get_fdata()
            name = i.split('_transformed.nii.gz')[0]
            corrected_pad_name=str(name)+'_corrected_pad.nii.gz'
            corrected_pad = np.subtract(img_data, bin_data)
            corrected_img=nib.Nifti1Image(corrected_pad,img.affine, img.header)
            nib.save(corrected_img, corrected_pad_name)




        avg_error = np.sum((np.abs(corrected_pad))) / np.count_nonzero(corrected_pad)
        str_avg = str(avg_error)
        #add path to text file
        with open('xxxxx', 'a') as f:
            f.write(str_avg)
            f.write('\n')
        print('mae error {:.4f}'.format(avg_error))
        total += avg_error

        #plotting
        pad = torch.from_numpy(corrected_pad)
        mask = torch.where(pad != 0, 1, 0)
        chr_tensor = torch.full_like(pad, a, dtype=torch.float32)
        chr_age = chr_tensor * mask
        # print(chr_age.shape, torch.unique(chr_age), chr_age)
        vox_pred = torch.add(pad, chr_age)
        # print(vox_pred.shape, torch.unique(vox_pred), vox_pred)
        vox_pred = vox_pred * mask
        # print(vox_pred.shape, torch.unique(vox_pred), vox_pred)
        vox_error = torch.sum(vox_pred) / torch.sum(mask)
        vox_preds.append(vox_error.item())

    plt.figure()
    # .scatter(chron_age, vox_preds)
    plt.xlim([(min(age) - 2), (max(age) + 2)])
    plt.ylim([(min(vox_preds) - 2), (max(vox_preds) + 2)])
    plt.xlabel('chronological age')
    plt.ylabel('pred age')
    plt.axline((0, 0), slope=1)
    for i, (vox, chr) in enumerate(zip(vox_preds, age)):
        plt.plot(chr, vox, 'o', color='red')
    #add path to save figure
    plt.savefig('xxxxxx')

    print('test set avg', total/len(files))

    test_results = total/len(files)
    str_test = str(test_results)

    #add path to same text file as before
    with open('xxxxx', 'a') as f:
        f.write('\n***test set avg****')
        f.write(str_test)

    

if __name__ == "__main__":
    test_before_correct()
    #test_after_correct()



