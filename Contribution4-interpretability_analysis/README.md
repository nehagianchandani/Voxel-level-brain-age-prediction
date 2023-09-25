## This directory contains the code pertaining to Contribution 4 - interpretability analysis

#### The global brain age prediction model has been adopted from [Gong et al., 2019](https://www.frontiersin.org/articles/10.3389/fpsyt.2021.627996/full) and [Peng et al., 2020](https://www.sciencedirect.com/science/article/pii/S1361841520302358).

The directory contains the following files:
* data_loader.py - creation of data loader for training/testing
* model.py - model definition
* run_train.py - script that initializes the model and runs the training script
* training.py - training script
* run_test.py - test script that generates interpretability heatmaps

### To train the global age prediction model
````
python run_train.py --batch_size 8 --learning_rate 0.001 --epochs 100 --results_dir 'PATH_TO_SAVE_RESULTS' --source_csv 'PATH_TO_CSV_WITH_REGISTERED_CAMCAN_DATA' --verbose True
````

### To test the global age prediction model and to generate the interpretability heatmaps
````
python run_test.py --results_dir 'PATH_TO_RESULTS_DIR' --source_csv 'PATH_TO_CSV_WITH_REGISTERED_CAMCAN_DATA' --model_path 'PATH_TO_SAVED_MODEL'
````

