import numpy as np
import pandas as pd
import nibabel as nib
import glob
import os


ctr =0
files = []
age_info=[]
#add path to prediction files
for filename in glob.iglob('xxxxxx', recursive=True):
    
    for f in os.listdir(filename):
        if f.endswith('transformed.nii.gz'):

            pad = os.path.join(filename, f)
            files.append(pad)
            age = int(f.split('_')[3])
            print(f, age)
            age_info.append(age)
        #details.append(text_file)
            ctr +=1

print(len(files))

print(len(age_info))
print(min(age_info), max(age_info))
print('----')

samples = files[:5]+files[25:]
samples_1 = files[5:25]
age = age_info[:5]+age_info[25:]
age_1 = age_info[5:25]
print(min(age), max(age))
values, counts = np.unique(age, return_counts=True)

print(len(samples_1), len(age_1))
print(len(samples), len(age))
testing = {'imgs': samples_1, 'age': age_1}
testing = pd.DataFrame(testing)
#add path to save destination for testing csv file
testing.to_csv('xxxxx', encoding='utf-8', index=False)

br = {'imgs': samples, 'age': age}
br = pd.DataFrame(br)
#add path to save destination for bias removal csv
br.to_csv('xxxxx', encoding='utf-8', index=False)


list_of_keys = ['img', 'age']



dictionary = ["age", "img"]
hierarchies = ["bin1", "bin2", "bin3", "bin4", "bin5", "bin6"]

bins = dict([(key, {'img':[], 'age':[]}) for key in hierarchies])


#create bins
for i, a in zip(samples, age):
    if a<45:
        bins['bin1']['img'].append(i)
        bins['bin1']['age'].append(a)
    elif a>= 45 and a<50:
        bins['bin2']['img'].append(i)
        bins['bin2']['age'].append(a)
    elif a>= 50 and a<55:
        bins['bin3']['img'].append(i)
        bins['bin3']['age'].append(a)
    elif a>= 55 and a<60:
        bins['bin4']['img'].append(i)
        bins['bin4']['age'].append(a)
    elif a>= 60 and a<65:
        bins['bin5']['img'].append(i)
        bins['bin5']['age'].append(a)
    elif a>= 65 and a<68:
        bins['bin6']['img'].append(i)
        bins['bin6']['age'].append(a)


#make array to save avg values at each voxel position in a bin
avg = np.zeros((182, 218, 182))


#for each bin in the bins dict, extract images list, stack them together in shape (#n, 121,145,121) where n is number of samples in each bin
#then for each stacked image, store avg value at each voxel 9across all images) in the avg array, then save the avg array
for key, val in bins.items():
    images = val['img']
    ages = val['age']
    print(len(images), len(ages))
    imgs=[]
    for i, name in enumerate(images):
        nifti = nib.load(name)
        img = nifti.get_fdata()
        imgs.append(img)
    imgs = np.stack(imgs)
    print(imgs.shape)
    avg = np.zeros((182, 218, 182))
    print(np.unique(avg))
    for a in range(182):
        for b in range(218):
            for c in range(182):
                avg[a,b,c]=np.mean(imgs[:,a,b,c])

    avg_img = nib.Nifti1Image(avg, nifti.affine)
    #add path for saving each bin file
    nib.save(avg_img, 'xxxxxx'+str(key))
    print('saved ', str(key))












