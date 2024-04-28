<div align="center">
    <h1>  <a href=https://openaccess.thecvf.com/content/WACV2024/papers/Sagar_MAdVerse_A_Hierarchical_Dataset_of_Multi-Lingual_Ads_From_Diverse_Sources_WACV_2024_paper.pdf>MAdVerse: A Hierarchical Dataset of Multi-Lingual Ads From Diverse Sources</a><br>
    <a href="https://www.python.org/downloads/release/python-31010/"><img src="https://img.shields.io/badge/python-3.10.10-blue"></a>
    <a href="https://pytorch.org/get-started/previous-versions/"><img src="https://img.shields.io/badge/made_with-pytorch 1.12.0-red"></a>
    <a href="https://zenodo.org/records/10657763"><img src="https://img.shields.io/badge/dataset-MAdVerse-pink"></a>
    <a href="https://madverse24.github.io/"><img src="https://img.shields.io/website?up_message=up&up_color=green&down_message=down&down_color=red&url=https://madverse24.github.io/"></a>
    <a href="https://youtu.be/j_NC3JxCVxs"><img src="https://badges.aleen42.com/src/youtube.svg"></a>
</div>

## :bookmark: Contents
1. [About](#information_source-about)
2. [Setting up the repository](#gear-setting-up-the-repository)
    1. [Create a virtual environment](#earth_asia-create-a-python-virtual-environment)
3. [Download](#arrow_down-download)
    1. [Dataset](#file_folder-dataset)
    2. [Hierarchical classifier checkpoints](#file_folder-hierarchical-classifier-checkpoints)
4. [Training your own hierarchical classifier](#desktop_computer-training-your-own-hierarchical-classifier)
5. [Ad NonAd Classifier](#desktop_computer-ad-nonad-classifier)
6. [Ad Detection](#desktop_computer-ad-detection)
7. [Ad Source Classification](#desktop_computer-ad-source-classification)
8. [Demo](#teacher-demo)
9. [Bibtex](#pushpin-bibtex)

## :information_source: About
This is the official code repository for WACV-2024 accepted paper "MAdVerse: A Hierarchical Dataset of Multi-Lingual Ads From Diverse Sources".The repository includes hierarchical classifiers, losses, and metrics for ad classification. We also provide code for extracting ad images from news paper, training Ad-NonAd classifier, deduplication of ads, and for tasks like source classificaiton, inducing hierarchy on other ad dataset and multilingual ads.
## :gear: Setting up the repository

### :earth_asia: Create a python virtual environment
1. This project requires python ```3.10.10```. Install it from <a href="https://www.python.org/downloads/release/python-31010/">python-31010</a>

2. Create a virtual environment using pip after downloading python

    * <b>Choose a directory for creating both the virtual environment and cloning the repository</b>
    ```
    >> cd /home/username/<preferred_path>
    ```
    * Create the virtual environment `madverse`.
    ```
    >> python3.10 -m pip install virtualenv
    >> python3.10 -m virtualenv madverse
    >> source madverse/bin/activate
    (madverse) >> pip install -r requirements.txt
    ```
3. Clone the repository.
    ```
    (madverse) >> git clone https://github.com/Amruth-sagar/MAdVerse.git
    (madverse) >> cd MAdVerse
    ```
    Run the following command, to install the package locally
    ```
    (madverse) >> pip install -e . 
    ```
    This installs the `hiercls` and its components locally.

## :arrow_down: Download
### :file_folder: Dataset
All the image folders and annotation files are hosted on **Zenodo**: **[MAdVerse](https://zenodo.org/records/10657763)**

Currently, the following are the available annotations for the images from these sources.

|                         | Online  | Advert Gallery | Newspaper  |
|:-----------------------:|:----------:|:--------------:|:-------------:|
| Hierarchical annotation |     YES    |       YES      |       NO      |
|     Source of the ad    |     YES    |       YES      |      YES      |
|    Language of the ad   |     YES    |       YES      |      YES      |
|     Brand / Product     |     YES    |       YES      |       NO      |


### :file_folder: Hierarchical classifier checkpoints
Here are the checkpoints for classifiers, top two performers from leaf-only and multilevel approaches. (Table-2 in paper)

| Architecture                      | Backbone | Loss               | Link |
|-----------------------------------|----------|--------------------|------|
| Leaf Only                         | BLIP-2   | Semantic embedding | [blip2-lo-BD](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/amruth_sagar_research_iiit_ac_in/Ef01PpaegrdFv-d5h9X6wL8BNo7aKqElg9gXsngW_EWIQw?download=1) |
| Leaf Only                         | BLIP-2   | Soft labels        | [blip2-lo-SL](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/amruth_sagar_research_iiit_ac_in/EfdP4Bt9w15Imh8iCcwtKkABIMpjWHkfy93-aZ196d4Oyg?download=1) |
| Multi level                       | BLIP-2   | Sum CE             | [blip2-ml-sCE](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/amruth_sagar_research_iiit_ac_in/Ea2y1RYmDZxDqr4zxunp3zIBuLO_NMn8-17ZCPZvzoDhsA?download=1) |
| Multi level (with feature fusion) | BLIP-2   | Sum CE             | [blip2-mlFF-sCE](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/amruth_sagar_research_iiit_ac_in/EeZgxjNM0KZBs_t0MW1I1acBK03K-GkiWCJomzkwPRt8ZQ?download=1) |


## :desktop_computer: Training your own hierarchical classifier
1. First step is to create train-val-test splits and other helper and annotation files, by running ```prepare_hier_annot_splits.py```. 
    
    For example, to create a 70-10-20 train-val-test split, with random seed as 1, for online ads, 
    ```
    (madverse) >> python prepare_hier_annot_splits.py \ 
                    --dataset_dir <path_to>/online_ads \
                    --train_split 70 \
                    --val_split 10 \
                    --test_split 20 \
                    --split_seed 1 \
                    --annot_dir <save_files_to>/madverse_annots
    ```

    This script will create all the files required for training and testing your own configuration.

2. Generate a configuration file similar to the provided one, and adjust the configuration parameters to align with your preferences.  

    The table below illustrates which architecture types, branch networks, and loss functions work together. Each row presents a valid configuration, where you choose one option from each cell in that row.

    <table>
    <thead>
    <tr>
        <th><b>Architecture<br></th>
        <th><b>Branches</th>
        <th <b>Losses</th>
    </tr>
    </thead>
    <tbody>
    <tr>
        <td><code>backbone_one_branch</code><br></td>
        <td><code>simple_branch</code></td>
        <td><code>simple_ce_l, barz_denzler_l, soft_labels_l, dot_l, hxe_l</code></td>
    </tr>
    <tr>
        <td><code>backbone_branch</code></td>
        <td><code>simple_branch</code></td>
        <td  rowspan="2"><code>sum_ce, dot</code></td>
    </tr>
    <tr>
        <td><code>backbone_branch_ff</code></td>
        <td><code>simple_branch_ff</code></td>
    </tr>
    </tbody>
    </table>

    The choices for backbones and metrics are
    
    <table >
    <thead>
    <tr>
        <th>
        <th>
    </tr>
    </thead>
    <tbody>
    <tr>
        <td><b>Backbones</td>
        <td><code>vit-base-patch16-224, vit-large-patch16-224-in21k, <br>convnext-base-224-22k, convnext-large-224-22k-1k, blip-v2</code></td>
    </tr>
    <tr>
        <td><b>Metrics</b></td>
        <td><code>accuracy, acc_top_2, acc_top_5, acc_top_10, <br>f1_score, precision, recall</code></td>
    </tr>
    <tr>
        <td><b>Hier Metrics</td>
        <td><code>lca_height_lw, lca_height_mistakes_lw, <br>tie_lw, tie_mistakes_lw</code></td>
    </tr>
    </tbody>
    </table> 

3.  For example, training a leaf-level-only classifier, with ConvNeXT-large as backbone, and hierarchical cross entropy as loss, you run
    ```
    (madverse) >> python train.py --config_path ./hiercls/config/nogit_config.yaml \
                    model.backbone.name="convnext-large-224-22k-1k" \
                    model.branch.name="simple_branch" \
                    model.meta_arch.name="backbone_one_branch" \
                    loss_class="hxe_l"
    ```
    <b>NOTE:</b> You can override the default choices in config file  by specifying any config variable with its option using dot notation, as demonstrated in the example.

4. For testing, select a checkpoint and execute the following. 
    ```
    (madverse) >> python test.py --config_path ./hiercls/config/config.yaml \
                    --ckpt_path <path_to>/ckpt.pth.tar \
                    --run_specific_config "" \
                    model.backbone.name="convnext-large-224-22k-1k" \
                    model.branch.name="simple_branch" \
                    model.meta_arch.name="backbone_one_branch" \
                    train.loss_class="hxe_l" 
                    test.results_path="test_results.xlsx"
    ```
    <b>NOTE:</b> If you want to predict at the parent level while using a leaf-only architecture, you can do that by setting *`test.bet_on_leaf_level=True`* in the configuration. This will give you all the parent labels corresponding to the predicted leaf-class for a given image. 

<br>
<br>

## :desktop_computer: Ad NonAd Classifier
Ads NonAds Classifier module provides functions for training the classifier, loading a pre-trained model, and making predictions on new input images. It also includes utility functions for preprocessing the input data and evaluating the performance of the classifier. More details can be found in the [Ad NonAd Classifier Readme](./ads_non_ads_classifier/README.md).

## :desktop_computer: Ad Detection
The Ad detection module contains scripts designed for training and executing an object detection model specialized in extracting advertisements from newspaper images. The annotations provided are customized to suit this particular task, aiming to identify and segment advertisements for further processing and analysis purposes. More information can be found in the [Ad Detection Readme](./ads_detector/Readme.md).

## :desktop_computer: Ad Source Classification
The Ad Source Classification module contains scripts for training and executing a model that classifies the source of an advertisement. The annotations provided are tailored to this task, with the goal of identifying the source of an advertisement for further analysis. More information can be found in the [Ad Source Prediction Readme](./ad_source_prediction/README.md).

## :teacher: Demo
We have provided a demo notebook [`demo.ipynb`](demo.ipynb) which loads a checkpoint to a model, and classifies a given Ad image.

#
## :pushpin: Bibtex
If you find our work or any part of this repository useful, please cite our paper!
```
@inproceedings{sagar2024madverse,
title = {{MAdVerse: A Hierarchical Dataset of Multi-Lingual Ads from Diverse Sources and Categories}},
author = {Amruth sagar and Rishabh Srivastava and Rakshitha and Venkata Kesav and Ravi Kiran},
booktitle = {IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
year = {2024}
}
```
