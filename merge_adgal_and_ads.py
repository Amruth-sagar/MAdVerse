from glob import glob
import os
import argparse
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import ipdb
from subprocess import run


# This script is for mergint ads from advert gallery, to online ads.


def get_brand_names(dataset_dir):
    brand_names = []

    for directory in os.scandir(dataset_dir):
        if directory.is_dir():
            sub_dirs = get_brand_names(directory.path)
            if not sub_dirs:
                brand_names.append(directory.path)
            else:
                brand_names.extend(sub_dirs)
    
    return brand_names

if __name__ == '__main__':
    argument_parser = argparse.ArgumentParser()
    
    argument_parser.add_argument('--adgal_path', help="Path of ad images from Advert Gallery", type=str, required=True)
    argument_parser.add_argument('--web_ads_path', help="Path of MAdVerse's web ads", type=str, required=True)

    arguments = argument_parser.parse_args()

    assert Path(arguments.adgal_path).is_dir(), "Given ad-gallery directory does not exists"
    assert Path(arguments.web_ads_path).is_dir(), "Given MAdVerse directory does not exists"

    adgal_paths = get_brand_names(arguments.adgal_path)
    web_ads_paths = get_brand_names(arguments.web_ads_path)

    initial_count_web_ads =  sum([len(os.listdir(path)) for path in web_ads_paths])
    initial_count_adgal = sum([len(os.listdir(path)) for path in adgal_paths])

    print("Number of images in web ads before merging: ", initial_count_web_ads)
    print("Number of images in advert gallery before merging: ", initial_count_adgal)
    print("{} + {} = {}".format(initial_count_web_ads, initial_count_adgal, initial_count_web_ads + initial_count_adgal))

    brands_in_adgal = [brand_path.split('/')[-1] for brand_path in adgal_paths]
    brands_in_web_ads = [brand_path.split('/')[-1] for brand_path in web_ads_paths]

    adgal_to_web_ads_mapping = defaultdict(str)

    # Creating a mapping between the source and destination brand folders
    # so that we can merge the images.
    for i, brand in enumerate(brands_in_adgal):
        try:
            index = brands_in_web_ads.index(brand)
            adgal_to_web_ads_mapping[adgal_paths[i]] = web_ads_paths[index]
        except:
            raise ValueError('{} does not exist in ad_gallery'.format(brand))
    
    for brand_path in adgal_paths:
        list_of_images = os.listdir(brand_path)
        destination_brand = adgal_to_web_ads_mapping[brand_path]

        source_paths = [brand_path + '/' + image_name for image_name in list_of_images]
        dest_paths = [destination_brand + '/' + image_name for image_name in list_of_images]
        
        for src, dest in tqdm(zip(source_paths, dest_paths)):
            run(['cp', src, dest])
        
    print("\nNumber of images after merging: ", sum([len(os.listdir(path)) for path in web_ads_paths]))


    

