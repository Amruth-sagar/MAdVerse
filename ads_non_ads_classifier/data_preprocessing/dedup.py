from imagededup.methods import CNN
from glob import glob
import argparse
from pathlib import Path
from tqdm import tqdm
import os
import shutil
shift =1

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

if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser()

    argument_parser.add_argument('--dataset_dir', help="Path of Ads dataset", type=str, required=True)
    argument_parser.add_argument('--min_sim_thresh', help="Similarity threshold", type=float, required=True)
    argument_parser.add_argument('--dup_res_file', help="Name of the text file to save the duplicate images", type=str, required=True)

    arguments = argument_parser.parse_args()
    
    duplicate_images = []
    image_encoder = CNN()
    
    dup_img_list = image_encoder.find_duplicates_to_remove(image_dir=arguments.dataset_dir, min_similarity_threshold=arguments.min_sim_thresh)
    duplicate_images.extend([x for x in dup_img_list])
    
    if shift:
        for images in tqdm(os.listdir(arguments.dataset_dir)):
            if images  in duplicate_images:
                os.remove(os.path.join(arguments.dataset_dir,images))
        

    with open(arguments.dup_res_file, 'w') as outfile:
        for image in duplicate_images:
            outfile.write(image + '\n')