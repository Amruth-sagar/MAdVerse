import os
from tqdm import tqdm
import argparse
argument_parser = argparse.ArgumentParser()

argument_parser.add_argument('--dataset_dir', help="Path of Ads dataset", type=str, required=True)
arguments = argument_parser.parse_args()

folder = arguments.dataset_dir

for images in tqdm(os.listdir(folder)):
    if images.endswith('~'):
        os.remove(os.path.join(folder,images))