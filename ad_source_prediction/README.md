
# Ad Source Classification
The Ad Source Classification module is a versatile deep learning system designed to categorize advertisements based on their source, offering flexibility for two primary classifications, namely Web Ads versus Newspaper Ads, or more granularly, Web Ads, Social Media Ads, and Newspaper Ads. 


## Table of Contents

- [Download Dataset](#download-dataset)
- [Download Pretrained Classifier](#download-pretrained-classifier)
- [Preprocessing](#preprocessing)
- [Training](#training)
- [Testing](#testing)
- [Classifying Images](#classifying-images)


## Load Dataset

You can download the dataset from this [link](https://iiitaphyd-my.sharepoint.com/:f:/g/personal/rishabh_s_students_iiit_ac_in/EmLx8rSmh0hDlp0E39UdgegBYLvWMPd3J5CDQ0DlC9XlBw?e=mEABRO).
The dataset consist of ad images categorized into two folders:
```
|------- two_class_dataset
    | 
    |--------TRAIN
    |        |----web_ads
    |            |----- image1.jpg
    |            |----- image2.jpg
    |            |----- ...
    |        |----newspaper_ads
    |            |----- image1.jpg
    |            |----- image2.jpg
    |            |----- ...
    |    |
    |    |----VAL
    |    |    |----web_ads
    |    |    |    |----- image1.jpg
    |    |    |    |----- image2.jpg
    |    |    |    |----- ...
    |    |    |----newspaper_ads
    |    |    |    |----- image1.jpg
    |    |    |    |----- image2.jpg
    |    |    |    |----- ...
    |    |
    |    |----TEST
    |         |----web_ads
    |         |    |----- image1.jpg
    |         |    |----- image2.jpg
    |         |    |----- ...
    |         |----newspaper_ads
    |              |----- image1.jpg
    |              |----- image2.jpg
    |              |----- ...
|----- three_class_dataset
    | 
    |--------TRAIN
    |        |----web_ads
    |            |----- image1.jpg
    |            |----- image2.jpg
    |            |----- ...
    |        |----social_media_ads
    |            |----- image1.jpg
    |            |----- image2.jpg
    |            |----- ...
    |        |----newspaper_ads
    |            |----- image1.jpg
    |            |----- image2.jpg
    |            |----- ...
    |    |
    |    |----VAL
    |    |    |----web_ads
    |    |    |    |----- image1.jpg
    |    |    |    |----- image2.jpg
    |    |    |    |----- ...
    |    |    |----social_media_ads
    |    |    |    |----- image1.jpg
    |    |    |    |----- image2.jpg
    |    |    |    |----- ...
    |    |    |----newspaper_ads
    |    |    |    |----- image1.jpg
    |    |    |    |----- image2.jpg
    |    |    |    |----- ...
    |    |
    |    |----TEST
    |         |----web_ads
    |         |    |----- image1.jpg
    |         |    |----- image2.jpg
    |         |    |----- ...
    |         |----social_media_ads
    |         |    |----- image1.jpg
    |         |    |----- image2.jpg
    |         |    |----- ...
    |         |----newspaper_ads
    |              |----- image1.jpg
    |              |----- image2.jpg
    |              |----- ...
|-----
```
## Download Pretrained Classifier

### :file_folder: Ad Source classifier checkpoints
Here are the checkpoints for classifiers for both three and two class source classification:

|Classification Type     | Backbone | Link |
|------------------------|----------|------|
|Three Class             | ViT      |[ViT](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/rishabh_s_students_iiit_ac_in/EXlEA8e1LmlDv3CBL8Vh6Q8BrM88KJIYwcQeQOYiEqxjqg?e=pTECGN) |
|Three Class             | Convnext |[Convnext](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/rishabh_s_students_iiit_ac_in/EWJ3JGpneelBtsHdfCtLRL0B53umenW_I4NX3nGQrl4m9g?e=V1tw9w) |
|Two Class               | ViT      |[ViT](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/rishabh_s_students_iiit_ac_in/EQEjYAnLocNLr_5BC3LSeKkBqrxvFHEBWSN739kaNwIk4g?e=69Zc1y) |
|Two Class             | Convnext |[Convnext](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/rishabh_s_students_iiit_ac_in/Ee1EgD7MDJ9KqEa8HCRzgh4B52ZKDWor8SjLgV3c09StlQ?e=zhZxt7) |


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