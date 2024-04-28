import json
from collections import defaultdict
from glob import glob
import os
import argparse
from pathlib import Path
import pickle
from math import ceil
import ipdb
import random

"""
    This script prepares hierarchical labels for images in the form of 
    [(image_path, 1,35,356), ...], and splits the annotation file itself, into TRAIN, VAL and TEST
    annotation files, such that, the dataset class has to load any one of these splits, and then it
    can direclty read the image using images_path, and also send hierarchical labels from __getitem__ .

    Along with these, it also saves level wise mapping between class names and integers.
    Ex: In level 0, it is like {'baby_products': 0, ...}

    NOTE: we also provide annotation files,
            >> web_annots_j.json (and also web_annots.pkl)
            >> adgal_annots_j.json (and also adgal_annots.pkl), 
            from which you can get the annotations of the images directly.
"""

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

def write_obj_to_file(object, filename):
    with open(filename, 'wb') as outfile:
        pickle.dump(object, outfile)

if __name__ == '__main__':
    argument_parser = argparse.ArgumentParser()
    
    argument_parser.add_argument('--dataset_dir', help="dataset path", type=str, required=True)
    argument_parser.add_argument('--train_split', help="percentage of train split", type=float, default=70)
    argument_parser.add_argument('--val_split', help="percentage of train split", type=float, default=10)
    argument_parser.add_argument('--test_split', help="percentage of test split", type=float, default=20)
    
    # Since our hierarchy varies in depth, you can balance it ( we merge the nodes such that all leaves are equidistant form root. (3 levels))
    # which will give 3-level annotations for all images. If you want all levels, use --imbalanced_hier.

    # NOTE: WE ONLY DEAL WITH BALANCED HIERARCHY IN CODE.
    argument_parser.add_argument('--balanced_hier', dest="is_balanced", action="store_true")
    argument_parser.add_argument('--imbalanced_hier', dest="is_balanced", action="store_false")
    argument_parser.set_defaults(is_balanced=True)

    argument_parser.add_argument('--split_seed', help="random seed for reproducability", type=int, default=1)
    argument_parser.add_argument('--annot_dir', help="path to save annotation and read other files", type=str, required=True)

    args = argument_parser.parse_args()

    assert args.train_split + args.val_split + args.test_split == 100, "Split percentages are not adding upto 100%"

    train_split = args.train_split / 100
    val_split = args.val_split / 100
    test_split = args.test_split / 100

    dataset_dir = Path(args.dataset_dir)
    assert dataset_dir.is_dir(), "Given dataset path does not exists"
    annot_dir = Path(args.annot_dir)
    os.makedirs(annot_dir.as_posix(), exist_ok=True)


    # Reading all the brands
    brand_names = get_brand_names(dataset_dir.as_posix())

    # Excluding MISC (Since they lack a definite leaf (brand))
    brand_names = [brand for brand in brand_names if 'MISC' not in brand]
    assert len(brand_names) == len(list(set(brand_names))), "Brands are not unique!"

    # Split the paths with '/' as delimiter, and get categories, sub_categories and brands
    paths_splitted = [brand.split('/') for brand in brand_names]

    # If a path name is /some/path/to/the/dataset/LEVEL1/LEVEL2/..../LEVELn, we will remove everyting upto dataset, 
    # and whatever remains are the name of the nodes at different levels.
    ignore_till_index = len(dataset_dir.as_posix().split('/'))

    paths_splitted = [words[ignore_till_index: ] for words in paths_splitted]

    if args.is_balanced:
        paths_splitted = [tuple(words[:3]) for words in paths_splitted]
        paths_splitted = list(set(paths_splitted))

    num_levels = max([len(x) for x in paths_splitted])

    print("Number of levels: ", num_levels)

    classes_in_each_level = defaultdict(list)

    for words in paths_splitted:
        for level, word in enumerate(words):
            classes_in_each_level[level].append(word)
            
    # Along with set(), we also sort the names of classes at each level. 
    classes_in_each_level = dict([(key, sorted(list(set(value)))) for key, value in classes_in_each_level.items()])

    for i in range(num_levels):
        print(f"Number of nodes in level {i+1}: ", len(classes_in_each_level[i]))

    int_to_str_mappings = []
    str_to_int_mappings = []

    for level, classes_in_curr_level in classes_in_each_level.items():
        int_to_str_mappings.append(dict([(i, class_name) for i, class_name in enumerate(classes_in_curr_level)]))
        str_to_int_mappings.append(dict([(class_name, i) for i, class_name in enumerate(classes_in_curr_level)]))

    for level, mapping in enumerate(int_to_str_mappings):
        with open(annot_dir / 'level_{}_int_to_str.pkl'.format(level), 'wb') as outfile:
            pickle.dump(mapping, outfile)

    for level, mapping in enumerate(str_to_int_mappings):
        with open(annot_dir / 'level_{}_str_to_int.pkl'.format(level), 'wb') as outfile:
            pickle.dump(mapping, outfile)  

    # Reading all the images paths
    all_img_paths = glob(dataset_dir.as_posix() + '/**/*.jpg', recursive=True)

    # MISC images are ignored as they lack a leaf node.
    all_img_paths = [path for path in all_img_paths if 'MISC' not in path]

    print("TOTAL IMG: ", len(all_img_paths))

    hier_annot_all_imgs = []

    for path in all_img_paths:
        words = path.split('/')
        
        # Here, the last string will be image name. Ignore that.
        words = words[ignore_till_index:-1]

        # folding the hierarchy = merging leaf nodes
        # which makes 1 -> 2 -> 3 -> 4 path  to 1 -> 2 -> 3, 
        # but keeps 1 -> 2 -> 3 as is.
        if args.is_balanced:
            words = words[:3]

        words = [str_to_int_mappings[level][word] for level, word in enumerate(words)]

        annotation = [path] + words
        hier_annot_all_imgs.append(annotation)

    # Storing the mappings as pkl files
    with open(annot_dir / 'hier_annot_ads.pkl', 'wb') as outfile:
        pickle.dump(hier_annot_all_imgs, outfile)

    train_annot = []
    val_annot = []
    test_annot = []

    # Setting the random seed before splitting
    random.seed(args.split_seed)

    for brand_path in brand_names:
        imgs_in_brands = os.listdir(brand_path)
        imgs_in_brands = [brand_path + '/' + img_path for img_path in imgs_in_brands]

        num_imgs_in_brands = len(imgs_in_brands)

        indices = list(range(0, num_imgs_in_brands))
        random.shuffle(indices)

        temp_annot = []
        for rand_idx in indices:
            path = imgs_in_brands[rand_idx]

            words = path.split('/')

            # Here, the last string will be image name. ignore that.
            words = words[ignore_till_index: -1]

            # Folding of hierarchy
            if args.is_balanced:
                words = words[:3]
            
            words = [str_to_int_mappings[level][word] for level, word in enumerate(words)]

            annotation = [path] + words
            temp_annot.append(annotation)
        
        train_idx = ceil(num_imgs_in_brands * train_split)
        val_idx = ceil(num_imgs_in_brands * (train_split + val_split))
        
        train_annot.extend(temp_annot[:train_idx])
        val_annot.extend(temp_annot[train_idx:val_idx])
        test_annot.extend(temp_annot[val_idx:])

    random.shuffle(train_annot)
    random.shuffle(val_annot)
    random.shuffle(test_annot)

    print("TRAIN: ", len(train_annot))
    print("VAL: ", len(val_annot))
    print("TEST: ", len(test_annot))

    with open(annot_dir / 'hier_annot_train.pkl', 'wb') as outfile:
        pickle.dump(train_annot, outfile)

    with open(annot_dir / 'hier_annot_val.pkl', 'wb') as outfile:
        pickle.dump(val_annot, outfile)
            
    with open(annot_dir / 'hier_annot_test.pkl', 'wb') as outfile:
        pickle.dump(test_annot, outfile)
    