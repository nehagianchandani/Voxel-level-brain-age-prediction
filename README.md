
# A Multitask Deep Learning Model for Voxel-level Brain Age Gap Estimation

#### Source code structure:
* Main repo contains the files for training
* Testing folder contains the testing source code with two csv (each for CamCAN & CC359 dataset) with test subjects id's
* Ablation experiment models use the same data-loader, the model architecture files are present in the ablation experiment folder. 

#### To download the data
* CAMCAN - [data access link](https://camcan-archive.mrc-cbu.cam.ac.uk/dataaccess/)
* CC359 - [data access link](https://docs.google.com/forms/d/e/1FAIpQLSe5hfUkyZQAFGP2yFKxEjv8h0KbIXyAKIHffwXCuQJ5Y7SqRw/viewform)


#### Step 1: To train the model

```
python main.py --learning_rate 0.001 --batch_size 8 --epochs 300 --root_dir #add root directory#
```
***

#### Step 2: To test the model

```
python main_test.py --root_dir #add root directory# --checkpoint_path #add path to saved model .pth file#

```

* The full paths in shuff_files_1.csv have been anonymized with xxxx excluding the name of input data files, make appropriate changes as per the location of files on your system.
* Paths for saving intermediary files like saved model, text file with results, images etc have been anonymized with xxxx for anonymity in various places across the code. Include the appropriate paths before running.
