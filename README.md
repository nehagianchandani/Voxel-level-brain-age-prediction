
# A Multitask Deep Learning Model for Voxel-level Brain Age Gap Estimation

## Source code structure:
* Main repo contains the files for training
* Testing folder contains the testing source code with two csv (CC359 & Camcan) test subjects (n=20 each)
* Bias Correction folder contains the source code for implementing bias correction as well as two csv. The bias_removal_imgs.csv has n=10 samples that were used for creating the bins and testing_imgs.csv (same as cc359_test.csv in testing directory) has n=20 samples to be used for final testing. 

### To download the data
* CC359 - [data access link](https://docs.google.com/forms/d/e/1FAIpQLSe5hfUkyZQAFGP2yFKxEjv8h0KbIXyAKIHffwXCuQJ5Y7SqRw/viewform)
* CAMCAN - [data access link](https://camcan-archive.mrc-cbu.cam.ac.uk/dataaccess/)

### To train the model

```
python main.py --learning_rate 0.001 --batch_size 8 --epochs 300 --root_dir #add root directory#
```
***

### To test the model

```
python main_test.py --root_dir #add root directory# --checkpoint_path #add path to saved model .pth file#

```
#### Additional changes:
* Add the appropriate path to csv file containing input data (shuff_files_1.csv) directly in  data_loader.py (for training and testing in the load_data() function)
* The full paths in shuff_files_1.csv have been anonymized with xxxx excluding the name of input data files, make appropriate changes as per the location of files on your system.
* Paths for saving intermediary files like saved model, text file with results, images etc have been anonymized with xxxx for anonymity in various places across the code. Include the appropriate paths before running.
