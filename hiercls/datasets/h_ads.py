from hiercls.utils.registry import registry
from numpy import asarray
import numpy as np
from pathlib import Path
import pickle
import ipdb
from glob import glob
from collections import defaultdict

def get_hierarchy(
    annotation_dir
):
    """Constructs hierarhcy by using the annotation files.

    Args:
        annotation_dir: directory where all the annotation related files are present
                        (NOTE: This works only if you have run 'prepare_hier_annot_splits.py', 
                        and hier_annot files of all splits and other files are created, and in annotation_dir.)
    """

    # The below variable contains per-level files, that are int-to-string maps.
    int_to_str_files = glob(annotation_dir + '/level_*_int_to_str.pkl')
    
    # Sorting the file names, such the during dictionary creation, correct level
    # number is mapped to correct file.
    int_to_str_files = sorted(int_to_str_files)

    int_to_str_files = [Path(mapping_file) for mapping_file in int_to_str_files]

    int_to_str_mappings = defaultdict(dict)

    for level, mapping_file in enumerate(int_to_str_files):
        with open(mapping_file, 'rb') as infile:
            int_to_str_mappings[level] = pickle.load(infile)


    all_images_hier_annot = annotation_dir / Path('hier_annot_ads.pkl')
    # Reading hierarchical annotation of the images.
    with open(all_images_hier_annot, 'rb') as infile:
        hier_annot = pickle.load(infile)

    # Here we take the annotations of all images (ignoring image names), and since
    # each annotation is a path from parent levels to leaf, a unique set of such 
    # paths is obtained, which is called hierarchy_int.
    hier_annot = [tuple(map(int, annot[1:])) for annot in hier_annot]
    hier_annot = list(set(hier_annot))

    hierarchy_int = asarray(hier_annot, dtype=int)

    hierarchy = []
    for annot in hier_annot:
        coarse_to_fine_names = []

        for level in range(0, len(annot)):
            coarse_to_fine_names.append(int_to_str_mappings[level][annot[level]])

        hierarchy.append(coarse_to_fine_names)

    return hierarchy, hierarchy_int
    


# NOTE: initially we created annotations if there were no files, by calling a 
# function from here. Later, we isolated that process as a one-time-run script,
# prepare_hier_annot_splits.py, which should be run first.

@registry.add_to_annot_creation_registry("hier_ads")
def create_hier_annot_allsplits(dataset_config):
    annotation_dir = dataset_config.annot_dir

    hierarchy, hierarchy_int = get_hierarchy(annotation_dir)

    return hierarchy, hierarchy_int
