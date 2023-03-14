
# A Multitask Deep Learning Model for Voxel-level Brain Age Gap Estimation

#### Source code structure:
* Main repo contains the files for training
* Testing folder contains the testing source code with two csv (each for CC359 & Camcan) with test subjects id's (n=20 each)
* Bias Correction folder contains the source code for implementing bias correction as well as two csv. The bias_removal_imgs.csv has n=10 samples that were used for creating the bins and testing_imgs.csv (same as cc359_test.csv in testing directory) has n=20 samples to be used for final testing. 

#### To download the data
* CC359 - [data access link](https://docs.google.com/forms/d/e/1FAIpQLSe5hfUkyZQAFGP2yFKxEjv8h0KbIXyAKIHffwXCuQJ5Y7SqRw/viewform)
* CAMCAN - [data access link](https://camcan-archive.mrc-cbu.cam.ac.uk/dataaccess/)

#### Step 1: To train the model

```
python main.py --learning_rate 0.001 --batch_size 8 --epochs 300 --root_dir #add root directory#
```
***

#### Step 2: To test the model

```
python main_test.py --root_dir #add root directory# --checkpoint_path #add path to saved model .pth file#

```

#### Step 3: Non-linear registration of PAD masks (using ANTs)
##### 3.1 - Register T1-weighted images to the MNI152 template provided in the 'templates' using the command
```
antsRegistrationSyN.sh -d 3 -f $FIXED -m $MOVING -o $OUTPUT -t 's' -p 'f'
```
$FIXED - MNI152 template  
$MOVING - T1-weighted MR image  
$OUPUT - suffix for output image  

##### 3.2 - Transform voxel-level PAD masks using the .mat file generated in 3.1 using the command
```
antsApplyTransforms -d 3 -i $MOVING -r $FIXED -o $OUTPUT --interpolation NearestNeighbor -t $TRANSFORM
```
$FIXED - MNI152 template  
$MOVING - voxel-level PAD mask  
$OUPUT - output file name  
$TRANSFORM - 0GenericAffine.mat file corresponding to each moving image generated in 3.1

#### Step 4: Bias Correction
* Use the registered PAD masks to perform bias correction. The bins corresponding to age groups are provided in the bias correction folder, you can skip the creating_bins.py files. Use the testing files (testing_imgs.csv) with the testing.py code (relevant function : test_after_correct()) to save corrected PAD masks.
* Make sure to replace xxxx with the correct paths.

#### Step 5: Inverse transform the voxel-level corrected PAD masks using the command
```
antsApplyTransforms -d 3 -i $MOVING -r $FIXED -o $OUTPUT --interpolation NearestNeighbor -t [$TRANSFORM,1]
```
$FIXED - original T1-weighted image  
$MOVING - voxel-level corrected PAD mask  
$OUTPUT -  output file name  
$TRANSFORM - 0GenericAffine.mat file corresponding to each moving image generated in 3.1  


#### Additional changes:
* Add the appropriate path to csv file containing input data (shuff_files_1.csv) directly in  data_loader.py (for training and testing in the load_data() function)
* The full paths in shuff_files_1.csv have been anonymized with xxxx excluding the name of input data files, make appropriate changes as per the location of files on your system.
* Paths for saving intermediary files like saved model, text file with results, images etc have been anonymized with xxxx for anonymity in various places across the code. Include the appropriate paths before running.
