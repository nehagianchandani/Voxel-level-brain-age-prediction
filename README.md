
# A Multitask Deep Learning Model for Voxel-level Brain Age Gap Estimation

## Source code structure:
* Main repo contains the files for training
* Testing folder contains the testing source code
* Bias Correction folder contains the source code for implementing bias correction

### To train the model

```
python main.py --learning_rate 0.001 --batch_size 8 --epochs 300 --root_dir #add root directory# --t1w_csv #add path to t1w path csv# --seg_mask_csv #add path to seg path csv# --age_csv_path #add path to age path csv# --brain_mask_csv #add path to brain binary masks path csv#

```
***

### To test the model

```
python main_test.py --root_dir #add root directory# --t1w_csv #add path to t1w path csv# --seg_mask_csv #add path to seg path csv# --age_csv_path #add path to age path csv# --brain_mask_csv #add path to brain binary masks path csv# --checkpoint_path #add path to saved model .pth file#

```
#### Additional changes:
* Add the appropriate path to csv file containing input data (shuff_files_1.csv) directly in  data_loader.py (for training and testing in the load_data() function) to skip --t1w_csv, --seg_masks_csv, age_csv, brain_mask_csv arguments in the command above.
* The full paths in shuff_files_1.csv have been anonymized with xxxx excluding the name of input data files, make appropriate changes as per the location of files on your system.
* Paths for saving intermediary files like saved model, text file with results, images etc have been anonymized with xxxx for anonymity in various places across the code. Include the appropriate paths before running.
