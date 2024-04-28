from hiercls.utils.registry import registry
from hiercls.datasets.datasets import *
from hiercls.models.meta_archs import *
from hiercls.utils.losses import *
from hiercls.utils.train_utils import *
from hiercls.utils.metrics import *
from hiercls.utils.helper_utils import *
from hiercls.datasets.data_utils import *
from hiercls.datasets.dataloader import *


import wandb
import torch
import argparse
import yaml
from omegaconf import OmegaConf


from torch.optim.adamw import AdamW
import ipdb


import pprint

# torch.autograd.set_detect_anomaly(True)

def read_and_merge_cfgs(arguments):
    config_path = arguments.config_path
    with open(config_path, "r") as infile:
        config = yaml.load(infile, Loader=yaml.loader.SafeLoader)

    config = OmegaConf.create(config)

    if arguments.opts:
        override_config = OmegaConf.from_dotlist(arguments.opts)
        config = OmegaConf.merge(config, override_config)

    return config


def parse_arguments():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="The name of the config file for this run.",
    )
    argument_parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Key-value pairs which will be read by omegaconf.",
    )
    return argument_parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()
    cfg = read_and_merge_cfgs(arguments)

    if cfg.wandb.use_wandb:
        wandb.init(project="<project_name>", entity="<entity_name>")

    seed_everything(cfg.seed)

    print(
        "========================================== CONFIG FOR THIS RUN =========================================="
    )
    pprint.pprint(OmegaConf.to_object(cfg))
    print(
        "=========================================================================================================\n\n"
    )

    print(
        "========================================== ARGS GIVEN IN THIS RUN =========================================="
    )
    pprint.pprint(arguments.opts)
    print(
        "=========================================================================================================\n\n"
    )

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    
    dataset_name = cfg.dataset.name
    dataset_config = cfg.dataset[dataset_name]
    dataset_config.seed = cfg.seed
    dataset_config.image_dimension = cfg.dataset.image_dimension

    
    # Getting unique paths in the hierarchy, which will be used to calculate many tree based metrics and of other 
    # things to work
    hierarchy, hierarchy_int = registry.mappings["annotation_creation"][dataset_name](
        dataset_config
    )

    (num_levels, num_cls_in_level) = get_stats_of_hierarchy(hierarchy_int)
    hierarchy_tree = get_nltk_tree_from_hierarchy(hierarchy_int)

    train_dataset = registry.mappings["datasets"][dataset_name](
        dataset_config, "train"
    )
    val_dataset = registry.mappings["datasets"][dataset_name](
        dataset_config, "val"
    )

    print("\n\nNumber of levels in hierarchy:", num_levels)
    print("Number of nodes in each level:", num_cls_in_level)

    hier_info = {"num_levels":num_levels, 
                "num_cls_in_level":num_cls_in_level,
                "hierarchy":hierarchy,
                "hierarchy_int": hierarchy_int,
                "hierarchy_tree":hierarchy_tree,
                "dataset_config":dataset_config}

    # hier_info is converted to a DotDict object, whose keys can be accessed by using dot notation
    # ex: 'hier_info.num_levels' is valid!    
    hier_info = DotDict(**hier_info)

    model = registry.mappings["meta_archs"][cfg.model.meta_arch.name](
        cfg.model,
        hier_info
    )

    # count_trainable_parameters(model)
    
    model.to(device)

    loss_config = cfg.loss_config[cfg.train.loss_class]
    criterion = registry.mappings["losses"][cfg.train.loss_class](loss_config, hier_info)
    criterion.to(device)
    
    # If we are using label embedding methods, we cannot use the argmax of probabilities to get predictions
    # we need to do similarity checking to get the predicted labels.
    loss_class = cfg.train.loss_class

    optimizers = [
        AdamW(
            params=model.parameters(),
            lr=cfg.train.lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=1e-5,
            amsgrad=True,
        )
    ]
    schedulers = None

    # The checkpoints will be saved in this directory,
    prefix_cfg_str = "data={};model={};loss={};seed={};batch={}".format(
        cfg.dataset.name,
        (cfg.model.backbone.name + "_" + cfg.model.branch.name),
        cfg.train.loss_class,
        cfg.seed,
        cfg.train.batch_size,
    )

    # To avoid cases where prefix_cfg_str can be same.
    prefix_cfg_str = prefix_cfg_str + cfg.wandb.unique_suffix

    if cfg.wandb.use_wandb:
        wandb.run.name = prefix_cfg_str

    train_dataloader = make_data_loader(
        dataset=train_dataset,
        batch_size=cfg.train.batch_size,
        num_workers=4,
        sampler=None,
    )
    val_dataloader = make_data_loader(
        dataset=val_dataset,
        batch_size=cfg.train.batch_size,
        num_workers=4,
        sampler=None,
    )

    prev_val_loss = 1e10

    if criterion.leaf_only:
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

    print(
        "========================================== TRAINING =========================================="
    )


    for i in range(cfg.train.epochs):
        train_loss, train_pred, train_gt, train_probs_epoch = train_one_epoch(
            train_dataloader=train_dataloader,
            criterion=criterion,
            model=model,
            curr_epoch=i,
            device=device,
            optimizers=optimizers,
            schedulers=None,
            loss_class=loss_class
        )

        train_metrics = calculate_metrics(cfg.metrics, train_pred, train_gt, num_levels, train_probs_epoch, loss_class)
        train_hier_metrics = calculate_hier_metrics(cfg.hier_metrics, train_pred, train_gt, tree_info_dict, num_levels, loss_class)

        val_loss, val_pred, val_gt, val_probs_epoch = valid_one_epoch(
            valid_dataloader=val_dataloader,
            criterion=criterion,
            model=model,
            curr_epoch=i,
            device=device,
            loss_class=loss_class
        )

        val_metrics = calculate_metrics(cfg.metrics, val_pred, val_gt, num_levels, val_probs_epoch, loss_class)
        val_hier_metrics = calculate_hier_metrics(cfg.hier_metrics, val_pred, val_gt, tree_info_dict, num_levels, loss_class)

        print("\nTrain loss: {}".format(train_loss))
        print("Val loss: {}".format(val_loss))

        if cfg.wandb.use_wandb:
            wandb.log({"loss/train": train_loss, "loss/val": val_loss}, step=i)


        log_metrics("Train Metrics", train_metrics, cfg.wandb.use_wandb, i)
        log_metrics("Train Hierarchical Metrics", train_hier_metrics, cfg.wandb.use_wandb, i)
        log_metrics("Val Metrics", val_metrics, cfg.wandb.use_wandb, i)
        log_metrics("Val Hierarchical Metrics", val_hier_metrics, cfg.wandb.use_wandb, i)

        if cfg.ckpt.save_ckpt is False:
            continue
        else:

            model_state_dict = model.state_dict()

            if prev_val_loss > val_loss:
                save_states = {
                    "epoch": i,
                    "model_state_dict": model_state_dict,
                }

                prev_val_loss = val_loss
                
                save_checkpoint(
                    state=save_states,
                    file_folder=cfg.ckpt.ckptdir,
                    experiment=prefix_cfg_str,
                    file_name="lowest_val_loss.pth.tar",
                )

            if (i + 1) % cfg.ckpt.ckptfreq == 0:
                save_states = {
                    "epoch": i,
                    "model_state_dict": model_state_dict,
                }
            
                save_checkpoint(
                    state=save_states,
                    file_folder=cfg.ckpt.ckptdir,
                    experiment=prefix_cfg_str,
                    file_name="epoch_{:03d}.pth.tar".format(i),
                ) 