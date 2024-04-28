import numpy as np
from glob import glob
import os
import shutil
import json
import argparse
from tqdm import tqdm
ratio = [0.9,0.08,0.02]
# seed = 42
seed =0
np.random.seed(seed)
argument_parser = argparse.ArgumentParser()
argument_parser.add_argument('--ads_dir', help="Path of Ads dataset", type=str, required=True)
argument_parser.add_argument('--non_ads_dir', help="Path of Non Ads dataset", type=str, required=True)
# h = hierarchical(folder/level1/level2/level3/image.jpg), nh = non-hierarchical(folder/image.jpg)
argument_parser.add_argument('--dataset_type', help="Type of dataset(h/nh)", type=str, required=True)
argument_parser.add_argument('--parent_folder', help="Storage location of the split", type=str, required=True)
arguments = argument_parser.parse_args()
spl=True # True for split, False for copy
for split in tqdm(['TRAIN','VAL','TEST']):
    if not os.path.exists(f'{arguments.parent_folder}/'+split):
        os.makedirs(f'{arguments.parent_folder}/'+split+'/ADS')
        os.makedirs(f'{arguments.parent_folder}/'+split+'/NON_ADS')
        
    if arguments.dataset_type=='h':
        ads =glob(f'{arguments.ads_dir}/*/*/*/*')
        non_ads = glob(f'{arguments.non_ads_dir}/*/*/*/*')
    else:
        ads =glob(f'{arguments.ads_dir}/*')
        non_ads = glob(f'{arguments.non_ads_dir}/*')
    np.random.shuffle(ads)
    np.random.shuffle(non_ads)
    if split=='TRAIN':
        ad = ads[:int(len(ads)*ratio[0])]
        non_ad = non_ads[:int(len(non_ads)*ratio[0])]
    if split=='VAL':
        ad = ads[int(len(ads)*ratio[0]):int(len(ads)*(ratio[0]+ratio[1]))]
        non_ad = non_ads[int(len(non_ads)*ratio[0]):int(len(non_ads)*(ratio[0]+ratio[1]))]
    if split=='TEST':
        ad = ads[int(len(ads)*(ratio[0]+ratio[1])):]
        non_ad = non_ads[int(len(non_ads)*(ratio[0]+ratio[1])):]
    for img in ad:
        shutil.copy(img,f'{arguments.parent_folder}/'+split+'/ADS') 
    for img in non_ad:
        shutil.copy(img,f'{arguments.parent_folder}/'+split+'/NON_ADS')   
                        
                