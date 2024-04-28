import numpy as np
from glob import glob
import os
import shutil
import json
import argparse
from tqdm import tqdm
num_classes = 3
ratio = [0.7,0.2,0.1]
seed = 1
np.random.seed(seed)
if num_classes ==2:
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('--class_1', help="Path of Ads dataset", type=str, required=True)
    argument_parser.add_argument('--class_2', help="Path of Non Ads dataset", type=str, required=True)
    # argument_parser.add_argument('--dataset_type', help="Type of dataset(h/nh)", type=str, required=True)
    argument_parser.add_argument('--parent_folder', help="Storage location of the split", type=str, required=True)
    arguments = argument_parser.parse_args()
    spl=True # True for split, False for copy
    for split in tqdm(['TRAIN','VAL','TEST']):
        if not os.path.exists(f'{arguments.parent_folder}/'+split):
            os.makedirs(f'{arguments.parent_folder}/'+split+'/WebAds')
            os.makedirs(f'{arguments.parent_folder}/'+split+'/NewsPaperAds')
            
            
        ads =glob(f'{arguments.ads_dir}/*/*/*/*')
        non_ads = glob(f'{arguments.non_ads_dir}/*/*/*')

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
        for img in tqdm(ad):
            shutil.copy(img,f'{arguments.parent_folder}/'+split+'/WebAds') 
        for img in tqdm(non_ad):
            shutil.copy(img,f'{arguments.parent_folder}/'+split+'/NewsPaperAds')  
elif num_classes ==3:
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('--class_1', help="Path of Web dataset", type=str, required=True)
    argument_parser.add_argument('--class_2', help="Path of Social dataset", type=str, required=True)
    argument_parser.add_argument('--class_3', help="Path of Newspaper dataset", type=str, required=True)
    argument_parser.add_argument('--parent_folder', help="Storage location of the split", type=str, required=True)
    arguments = argument_parser.parse_args()
    spl=True # True for split, False for copy
    for split in tqdm(['TRAIN','VAL','TEST']):
        if not os.path.exists(f'{arguments.parent_folder}/'+split):
            os.makedirs(f'{arguments.parent_folder}/'+split+'/WebAds')
            os.makedirs(f'{arguments.parent_folder}/'+split+'/SocialMediaAds')
            os.makedirs(f'{arguments.parent_folder}/'+split+'/NewsPaperAds')
            
        web_ads =glob(f'{arguments.class_1}/*/*/*/*')
        social_ads = glob(f'{arguments.class_2}/*')
        newspaper_ads = glob(f'{arguments.class_3}/*/*/*')

        np.random.shuffle(web_ads)
        np.random.shuffle(social_ads)
        np.random.shuffle(newspaper_ads)
        if split=='TRAIN':
            web_ad = web_ads[:int(len(web_ads)*ratio[0])]
            social_ad = social_ads[:int(len(social_ads)*ratio[0])]
            newspaper_ad = newspaper_ads[:int(len(newspaper_ads)*ratio[0])]
        if split=='VAL':
            web_ad = web_ads[int(len(web_ads)*ratio[0]):int(len(web_ads)*(ratio[0]+ratio[1]))]
            social_ad = social_ads[int(len(social_ads)*ratio[0]):int(len(social_ads)*(ratio[0]+ratio[1]))]
            newspaper_ad = newspaper_ads[int(len(newspaper_ads)*ratio[0]):int(len(newspaper_ads)*(ratio[0]+ratio[1]))]
        if split=='TEST':
            web_ad = web_ads[int(len(web_ads)*(ratio[0]+ratio[1])):]
            social_ad = social_ads[int(len(social_ads)*(ratio[0]+ratio[1])):]
            newspaper_ad = newspaper_ads[int(len(newspaper_ads)*(ratio[0]+ratio[1])):]
        for img in tqdm(web_ad):
            shutil.copy(img,f'{arguments.parent_folder}/'+split+'/WebAds') 
        for img in tqdm(social_ad):
            shutil.copy(img,f'{arguments.parent_folder}/'+split+'/SocialMediaAds')  
        for img in tqdm(newspaper_ad):
            shutil.copy(img,f'{arguments.parent_folder}/'+split+'/NewsPaperAds')
    
     
                            
                