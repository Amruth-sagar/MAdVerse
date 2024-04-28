from glob import glob
dataset_folder = '/ssd_scratch/cvit/rishabh.s/Filtered_ads'

ads = glob(dataset_folder + '/ads/*')
non_ads = glob(dataset_folder + '/non-ads/*')
 
# Create train, val test folders and distribute ads and non-ads with ration 80:10:10
import os
import shutil
import random

random.seed(0)














