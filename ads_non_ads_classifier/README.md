# Ads-Non Ads Classifier
This is the ad_non-ad classifier module.

The ad_non-ad classifier is a deep learning model that is trained to classify whether a given input is an advertisement or not. By analyzing features such as color, texture, and shape, the classifier learns to identify patterns specific to advertisements.  
This module provides functions for training the classifier, loading a pre-trained model, and making predictions on new inputs. It also includes utility functions for preprocessing the input data and evaluating the performance of the classifier.



## Table of Contents

- [Download Dataset](#download-dataset)
- [Download Pretrained Classifier](#download-pretrained-classifier)
- [Preprocessing](#preprocessing)
- [Training](#training)
- [Testing](#testing)
- [Classifying Images](#classifying-images)


## Download Dataset

You can download the dataset from [zenodo_link](http://dataset_url).
The dataset is divided in four folders:
- Online Ads 
- Advert_Gallery
- Epaper1
- Epaper2


## Download Pretrained Classifier

### :file_folder: Ad Non Ad Classifier
The best checkpoint of Ad Non Ad Classifier(ViT Backbone) can be downloaded from here:
[Ads-Non-Ads-Classifier](https://iiitaphyd-my.sharepoint.com/:f:/g/personal/rishabh_s_students_iiit_ac_in/Ep0kFw8Il-pDsrVs-lmUhmEBDbPi7Y4s59u2xMJiN2jpwg?e=BwkQwj)

## Preprocessing
There are four major steps in preprocessing the data:
1. **fix_mogrify.py**: This file is used to remove bad png images, courrupted images, etc. in the folder. It can be run by ```python fix_mogrify.py --dataset_dir <input_dir>``` where <input_dir> is the path to the folder containing images.
2. **dedup.py**: This file is used to find and remove duplicate images in the folder. It can be run by ```python dedup.py --dataset_dir <input_dir> ---min_sim_thresh <similarity_threshold> --dup_res_file <results_pkl_file>``` where <input_dir> is the path to the folder containing images, <similarity_threshold> is the thresold below which image should be consider as same(~0.9) and <results_pkl_file> is output file to store the result of deduplication.
3. **handle_extension.py**: This file is used to handle the images with different extensions and convert all images in jpeg format. It can be run by ```python handle_extension.py --dataset_dir <input_dir>``` where <input_dir> is the path to the folder containing images.
4. **CreateSplit.py** This file is used to create the train, val and test split of the dataset. It can be run by ```python CreateSplit.py --ads_dir <path_to_ads_dataset> --non_ads_dir <path_to_non_ads_dataset> --dataset_type <h/nh> --parent_folder <path_to_parent_folder>``` where

- `--ads_dir`: Path to the directory containing advertisement images.
- `--non_ads_dir`: Path to the directory containing non-advertisement images.
- `--dataset_type`: Type of dataset. Use 'h' for hierarchical (folder/level1/level2/level3/image.jpg) or 'nh' for non-hierarchical (folder/image.jpg).
- `--parent_folder`: Storage location of the split.

## Training

To train the classifier, follow these steps:

1. Modify the configuration file `config.yaml` according to your requirements, change the train_dir and val_dir parameters.
2. Run the training script:
   ```bash
   python train.py
    ```

### Testing
To test a classifier checkpoint on the test dataset, follow these steps:

1. Modify the configuration file `config.yaml` according to your requirements, change the train_dir and val_dir parameters.

2. Run the training script:
   ```bash
    python test.py --ckpt_path <checkpoint_path> --pred_out <prediction file path>

    ```
    Replace <checkpoint_path> with the path to the checkpoint file and <prediction_file_path> with the desired path to save the predictions
### Classifying Images
To classify images using a pre-trained classifier, follow these steps:

1. Modify the configuration file `config.yaml` according to your requirements, change the ads and non_ads parameters.

2. Run the training script:
   ```bash
     python classify.py --dataset <image folder> --ckpt_path <checkpoint_path> --pred_out <prediction file path> --out_folder <folder to store classified ads>


    ```
    Replace <image folder> with the path to the folder containing images, <checkpoint_path> with the path to the checkpoint file, < prediction file path > with the desired path to save the predictions, and < folder to store classified ads > with the path to the folder where classified ads will be stored.