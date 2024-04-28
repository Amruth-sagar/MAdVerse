import os
import argparse
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import ipdb
import pickle
from imagededup.methods import CNN
import random


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
    
    argument_parser.add_argument('--adriana_path', help="Path where Pitts's Ad dataset is located", type=str, required=True)
    argument_parser.add_argument('--madverse_path', help="Path of the MAdVerse dataset", type=str, required=True)
    argument_parser.add_argument('--sampling_seed', help="Seed for sampling and reproducibility", type=int, required=True)
    argument_parser.add_argument('--num_images', help="Number of images to sample", type=int, required=True)
    argument_parser.add_argument('--sim_thresh', help="Similarity threshold to delete images", type=float, default=0.95)
    argument_parser.add_argument('--dup_file_path', help="Path to save duplicate file to", type=str, default=None)


    arguments = argument_parser.parse_args()

    assert Path(arguments.adriana_path).is_dir(), "Given adriana ad directory does not exists"
    assert Path(arguments.madverse_path).is_dir(), "Given MAdVerse directory does not exists"

    adriana_ad_image_paths = os.listdir(arguments.adriana_path)
    adriana_ad_image_paths = [arguments.adriana_path +'/'+ path for path in adriana_ad_image_paths]

    brand_paths = get_brand_names(arguments.madverse_path)

    # Setting the random seed and randomly selecting the images.
    random.seed(arguments.sampling_seed)

    # Randomly selecting images (filenames) from Pitt's Ad dataset
    randomly_selected_paths = random.sample(adriana_ad_image_paths, arguments.num_images)


    

    #===============#
    # DEDUPLICATION #
    #===============# 

    model = CNN()

    # Get embeddings for all sampled ad images
    path_embedding_mapping_of_pitts_ads = defaultdict()
    for path in adriana_ad_image_paths:
        path_embedding_mapping_of_pitts_ads[path] = model.encode_image(image_file=path)


    # Now lets compare these embeddings, with the embeddings of images for all brands. 
    # (on average 30 imgs/brand ), and we store these to a dictionary.
    duplicate_images = dict()
    for path in tqdm(brand_paths):
        brand_embedding_mapping = model.encode_images(image_dir=path)
        merged_mapping = {**path_embedding_mapping_of_pitts_ads, **brand_embedding_mapping}
        
        dup_images = model.find_duplicates(encoding_map=merged_mapping, min_similarity_threshold=arguments.sim_thresh)
        duplicate_images.update(dup_images)


    # This dictionary which has entries in the form 
    # {
    # 'image1.jpg': ['image1_duplicate1.jpg',
    #                 'image1_duplicate2.jpg'],
    # 'image2.jpg': [..],
    # ..
    # }
    # will be saved. 
    if arguments.dup_file_path:
        with open(arguments.dup_file_path + '/duplicates.pkl','wb') as outfile:
            pickle.dump(duplicate_images, outfile)