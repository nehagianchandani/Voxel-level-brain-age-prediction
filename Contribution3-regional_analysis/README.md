## Contribution 3 - Regional Analysis of voxel-level brain age


_**It is highly recommended to go through all code and csv files before running any scripts and replacing any placeholders (XXXX) with appropriate paths.**_

#### Directory structure:
* MNI-maxprob-thr50-1mm.nii.gz - MNI regional atlas (9 regions)
* MNI152_T1_1mm_brain.nii - MNI brain template 
* adni_diseased_subs_meta.csv and oasis_diseased_subs_meta.csv - csv files corresponding to the two diseased test sets
* regional_age_avgs_image_native_space.py - script to get average regional values

#### To download the data
* OASIS - [data access link](https://www.oasis-brains.org/)
* ADNI - [data access link](https://adni.loni.usc.edu/data-samples/access-data/)

### To register the MNI regional atlas to each T1w images's native space
#### Step 1: Register the MNI brain atlas to each image's native space
```
sh commands_to_sbatch_registraton.sh
```
***

#### Step 2: Use the mat file generated in step 1 to and transform the MNI regional atlas to image space
```
sh commands_to_sbatch_transformation.sh

```

### Once step 1 and step 2 are successful, the following command can be used to get the regional PAD average and SD for the test sets:
```
python regional_age_avgs_image_native_space.py

```
