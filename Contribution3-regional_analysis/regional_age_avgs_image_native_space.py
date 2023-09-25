import nibabel as nib
import numpy as np
import glob
import pandas as pd
import os

# to get regional PAD averages (in each MRI's native space), run this code.
# to run for ADNI - alzheimer's test set, comment lines 44-73
# to run for OASIS - dementia test set, comment lines 11-42

# # for ADNI - alzheimer's test set
#replace XXXX with path to directory with PADs, T1w images, bias corrected PAD maps and MNI regional atlas registered to image space is saved. - ADNI
pads = sorted(glob.glob(r"XXXX/*/*_corrected_invtransformed2.nii.gz")) 
atlases = sorted(glob.glob(r"XXXX/*/*MNI_brain.nii.gz"))
# for adni diseased young
print('adni diseased')
print(len(pads), len(atlases))


# for adni diseased - replace XXXX with path to csv
meta = pd.read_csv(r"XXXX/adni_diseased_subs_meta.csv")
age=[]
for i in pads:
    name = i.split('/')[-1].split('_corrected_invtransformed2.nii.gz')[0] +".nii.gz"
    if name in meta['imgs'].str.split('/').str[-1].tolist():
        a = meta.loc[meta['imgs'].str.split('/').str[-1] == name, 'age'].values[0]
        age.append(int(a))
pads2=[]
age2=[]
atlases2=[]

for i, age_val, a in zip(pads, age, atlases):
    if age_val<71:
            pads2.append(i)
            atlases2.append(a)
            age2.append(age_val)

print('filtered for age',len(pads2), len(atlases2))

pads = pads2
atlases=atlases2
age = age2

# # for oasis - DEMENTIA test set
# #replace XXXX with path to directory with PADs, T1w images, bias corrected PAD maps and MNI regional atlas registered to image space is saved. - OASIS
# pads = sorted(glob.glob(r"XXXX/*/*_corrected_invtransformed2.nii.gz"))
# atlases = sorted(glob.glob(r"XXXX/*/*MNI_brain.nii.gz"))
# print('oasis diseased')
# print(len(pads), len(atlases))

# # for oasis diseased - replace XXXX with path to csv
# meta = pd.read_csv(r"/XXXX/oasis_diseased_subs_meta.csv")
# age=[]
# for i in pads:
#     name = i.split('/')[-1].split('_corrected_invtransformed2.nii.gz')[0] +".nii.gz"
#     if name in meta['imgs'].str.split('/').str[-1].tolist():
#         a = meta.loc[meta['imgs'].str.split('/').str[-1] == name, 'age'].values[0]
#         age.append(int(a))
    
# pads1=[]
# atlases1=[]
# age1=[]

# for i, a, age_val in zip(pads, atlases, age):
#     if age_val<71:
#         pads1.append(i)
#         atlases1.append(a)
#         age1.append(age_val)
# print('filtered for age', len(pads1), len(atlases1), len(age1))

# pads = pads1
# atlases=atlases1
# age = age1


print('\n')


labels = ['zero vals', 'Caudate', 'Cerebellum', 'Frontal Lobe', 'Insula','Occipital Lobe','Parietal Lobe','Putamen', 'Temporal Lobe', 'Thalamus']         
avg_dict = {new_list: [] for new_list in np.arange(0,10,1, dtype=np.float32)}
std_dict = {new_list: [] for new_list in np.arange(0,10,1, dtype=np.float32)}
ctr = 0
counter = 0
for img, atlas, a in zip(pads, atlases, age):
    img_d = nib.load(img).get_fdata()
    atlas_d = nib.load(atlas).get_fdata()
    counter = counter +1
    shape = img_d.shape
    ctr = ctr +1

    print(f"{ctr} sample : {img.split('/')[-1]}", flush=True)
    img_avg_dict = {new_list: [] for new_list in np.arange(0,10,1, dtype=np.float32)}
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                if img_d[i,j,k] != 0.:
                    img_avg_dict[atlas_d[i,j,k]].append(img_d[i,j,k])

    
    for i in np.arange(0,10,1, dtype=np.float32):
        print(f"{i} (age: {a}): {labels[int(i)]} : {np.average(img_avg_dict[i])} \u00B1 {np.std(img_avg_dict[i])}", flush=True)
        avg_dict[i].append(np.average(img_avg_dict[i]))
        std_dict[i].append(np.std(img_avg_dict[i]))
    print('----------------------------------------------', flush=True)


print('----------------------------------------------', flush=True)
print('OVERALL AVG IN RAW IMG SPACE: ', flush=True)
for i in np.arange(0,10,1, dtype=np.float32):
        print(f"{i} {labels[int(i)]} :  {np.average(avg_dict[i])} \u00B1 {np.std(avg_dict[i])}", flush=True)

print('\n\nOVERALL STD IN RAW IMG SPACE: ', flush=True)
for i in np.arange(0,10,1, dtype=np.float32):
        print(f"{i} {labels[int(i)]} :  {np.average(std_dict[i])} \u00B1 {np.std(std_dict[i])}", flush=True)
