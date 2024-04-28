from hiercls.utils.registry import registry
from hiercls.datasets.datasets import *
from hiercls.models.meta_archs import *
from hiercls.utils.losses import *
from hiercls.utils.train_utils import *
from hiercls.utils.metrics import *
from hiercls.utils.helper_utils import *
from hiercls.datasets.data_utils import *
from hiercls.datasets.dataloader import *

import numpy as np
import pandas as pd

import torch
import argparse
import yaml
from omegaconf import OmegaConf
from pathlib import Path



from pprint import PrettyPrinter

def read_and_merge_cfgs(arguments):
    config_path = arguments.config_path
    with open(config_path, 'r') as infile:
        config = yaml.load(infile, Loader=yaml.loader.SafeLoader)

    config = OmegaConf.create(config)

    if arguments.opts:
        override_config = OmegaConf.from_dotlist(arguments.opts)
        config = OmegaConf.merge(config, override_config)

    return config 


def parse_arguments():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('--config_path', type=str, required=True, help="The name of the config file for this run.")
    argument_parser.add_argument('--ckpt_path', type=str, required=True, help="path to the checkpoint file.")
    argument_parser.add_argument('opts', default=None, nargs=argparse.REMAINDER, help="Key-value pairs which will be read by omegaconf.")
    return argument_parser.parse_args()

if __name__=='__main__':
    arguments = parse_arguments()
    cfg = read_and_merge_cfgs(arguments)

    seed_everything(cfg.seed)

    print("========================================== CONFIG FOR THIS RUN ==========================================")
    PrettyPrinter(indent=4).pprint(OmegaConf.to_object(cfg))
    print("=========================================================================================================\n\n")

    print("========================================== ARGS GIVEN IN THIS RUN ==========================================")
    PrettyPrinter(indent=4).pprint(arguments)
    print("=========================================================================================================\n\n")   

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    dataset_name = cfg.dataset.name
    dataset_config = cfg.dataset[dataset_name]
    dataset_config.seed = cfg.seed
    dataset_config.image_dimension = cfg.dataset.image_dimension

    # Getting unique paths in the hierarchy, which will be used to calculate many tree based metrics and of other 
    # things to work
    hierarchy, hierarchy_int = registry.mappings['annotation_creation'][dataset_name](dataset_config)

    test_dataset = registry.mappings['datasets'][dataset_name](dataset_config, "test")

    num_levels, num_cls_in_level = get_stats_of_hierarchy(hierarchy_int)
    hierarchy_tree = get_nltk_tree_from_hierarchy(hierarchy_int)


    # hier_info contains all the information about the hierarchy.
    # It is converted to DotDict, whose keys can be accessed using dot notation
    hier_info = {"num_levels":num_levels, 
                "num_cls_in_level":num_cls_in_level,
                "hierarchy":hierarchy,
                "hierarchy_int": hierarchy_int,
                "hierarchy_tree":hierarchy_tree,
                "dataset_config":dataset_config}
    hier_info = DotDict(**hier_info)

    model = registry.mappings["meta_archs"][cfg.model.meta_arch.name](
        cfg.model,
        hier_info
    )

    ckpt = torch.load(arguments.ckpt_path)

    # POSTHOC METHODS LIST: Initially we planned on applying post-hoc methods
    # which changes the softmax probabilities to reduce severity of mistakes, 
    # but couldn't advance in that direction.
            
    # For that reason, keep 'posthoc_methods_list' as None
    posthoc_names_list = []
    posthoc_methods_list = None

    # Loading the checkpoint.
    # Disabling the strict mode
    model.load_state_dict(ckpt["model_state_dict"])

    del(ckpt)
    model.to(device)

    loss_config = cfg.loss_config[cfg.train.loss_class]
    criterion = registry.mappings["losses"][cfg.train.loss_class](loss_config, hier_info)
    criterion.to(device)

    loss_class = cfg.train.loss_class

    print("========================================== TESTING ==========================================")

    test_dataloader = make_data_loader(dataset=test_dataset, batch_size=cfg.train.batch_size, num_workers=4, sampler=None)

    # When you classify only leaf nodes but want parent-level predictions too, 
    # set 'bet_on_leaf_level` = TRUE in config. This will treat leaf predictions as given, 
    # and provides predictions for each parent level based on the path from predicted leaf to root.
        
    test_loss, test_pred, test_gt, test_probs_epoch = test_one_epoch(
        test_dataloader = test_dataloader,
        criterion = criterion,
        model = model,
        device = device,
        hierarchy_int = hierarchy_int,
        posthoc_methods_list = posthoc_methods_list,
        bet_on_leaf_level = cfg.test.bet_on_leaf_level,
        loss_class = loss_class
    )

    if criterion.leaf_only:
        if cfg.test.bet_on_leaf_level:
            tree_info_dict = {
            "lca_heights" : get_level_wise_lca_heights(hierarchy_int, dataset_config),
            "tie_distances" : get_level_wise_tie_distances(hierarchy_int, dataset_config),
            }
        else:
            tree_info_dict = {
                "int_to_node" : criterion.int_to_node,
                "lca_heights" : get_lca_height_of_leaves(hierarchy_int, dataset_config),
                "tie_distances" : get_tie_of_leaves(hierarchy_int, dataset_config),
            }
    else:
        tree_info_dict = {
            "lca_heights" : get_level_wise_lca_heights(hierarchy_int, dataset_config),
            "tie_distances" : get_level_wise_tie_distances(hierarchy_int, dataset_config),
        }

    tree_info_dict = DotDict(**tree_info_dict)


    test_metrics = calculate_metrics(cfg.metrics, test_pred, test_gt, num_levels, test_probs_epoch, loss_class, criterion=criterion, bet_on_leaf=cfg.test.bet_on_leaf_level)
    test_hier_metrics = calculate_hier_metrics(cfg.hier_metrics, test_pred, test_gt, tree_info_dict, num_levels, loss_class)

    print("test loss: {}".format(test_loss))

    all_metrics_data = []


    log_metrics(metric_group="Test Metrics", metric_values=test_metrics, log_to_wandb=False, step_number=0)
    log_metrics(metric_group="Test Hierarchical Metrics", metric_values=test_hier_metrics, log_to_wandb=False, step_number=0)
    
    print("\n\n\n")


    for key, value in test_metrics.items():
        curr_metric_values = [x[1] for x in value]
        all_metrics_data.append(curr_metric_values)

    for key, value in test_hier_metrics.items():
        curr_metric_values = [x[1] for x in value]
        all_metrics_data.append(curr_metric_values)

    # Left padding irregular rows with np.nan
    num_cols_per_row = []
    for values in all_metrics_data:
        num_cols_per_row.append(len(values))

    max_row_length = max(num_cols_per_row)

    padded_data = [
        [np.nan] * (max_row_length - len(row)) + row  
        for row in all_metrics_data
    ]

    all_metrics_data = padded_data



#================================================================================================#
#               Writing results to excel sheet, with hierarchical rows and columns.              #
#================================================================================================#

    # All the metrics will be rounded to 5 decimal points. 
    # This should be done in order to avoid duplicate entries to the hierarchical file. 
    all_metrics_data = np.round(np.asarray(all_metrics_data, dtype=float), 5)

    result_xlxs_path = Path(cfg.test.results_path)
    result_pkl_path = Path(cfg.test.results_path[:-5] + '.pkl')

    
    if result_pkl_path.is_file():
        with open(result_pkl_path, 'rb') as infile:
            previous_data = pickle.load(infile)
    
            
    
    dimensions = ['dataset', 'meta_arch', 'loss', 'metrics', 'levels']

    # Here the 'levels' depend on the number of branches in the architecture.   
    coords = {
        'dataset':[cfg.dataset.name], 
        'meta_arch':[cfg.model.backbone.name+'__'+cfg.model.branch.name],
        'loss':[cfg.train.loss_class],
        'metrics': cfg.metrics + cfg.hier_metrics,
        'levels': ['leaf_level'] if criterion.leaf_only and not cfg.test.bet_on_leaf_level else ['level_1', 'level_2', 'level_3']
        }
    multi_dimensional_index = pd.MultiIndex.from_product([coords[dimensions[i]] for i in range(len(dimensions))], names=dimensions)
    data_frame = pd.Series(all_metrics_data.flatten(), index=multi_dimensional_index)

    # This unstacks the metrics and levels, such that the columns are metrics, and the sub columns are the levels 
    data_frame = data_frame.unstack(level=[-2,-1])

    print(data_frame)

    # If the file already exixts, then append the current metrics to the previous ones.
    if result_pkl_path.is_file():
        data_frame = pd.concat([previous_data, data_frame])

    # This is to remove duplicate entries caused by testing the same thing twice.
    data_frame = data_frame.drop_duplicates()

    with pd.ExcelWriter(result_xlxs_path) as writer:
        data_frame.to_excel(writer)

    with open(result_pkl_path, 'wb') as outfile:
        pickle.dump(data_frame, outfile)
    
