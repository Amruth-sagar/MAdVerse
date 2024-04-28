import os
import torch
import yaml
from prettytable import PrettyTable
import random
import numpy as np
import wandb


def save_checkpoint(
    state, file_folder, experiment, file_name="checkpoint.pth.tar"
):
    """save checkpoint to file"""

    file_folder = os.path.join(file_folder, experiment)
    os.makedirs(file_folder, exist_ok=True)
    torch.save(state, os.path.join(file_folder, file_name))


def load_config(config_file):
    with open(config_file, "r") as fd:
        config = yaml.load(fd, Loader=yaml.FullLoader)
    return config


def log_metrics(metric_group, metric_values, log_to_wandb, step_number):
    """
    ------------------------------------------------------
    Args:
        metric_group: a string that tells what metrics they are
                        like "Train metrics" or "Test metrics"
                        etc
        metric_values: A nested dictionary, with metric name as
                        key and value is another dict, which has
                        multiple levels.
        log_to_wandb: if True, the above metrics will be logged to 
                        wandb
    """
    print("\n" + metric_group)
    for key, value in metric_values.items():
        print('"{}:\n\t{}"'.format(key, value))

        if log_to_wandb:
            for val in value:
                wandb.log(
                    {"{}/{}".format(key, val[0]): val[1]}, step=step_number
                )

def count_trainable_parameters(model):
    table = PrettyTable(["Modules", "Requires grad", "Trainable parameters"])
    table.align["Modules"] = "l"
    table.align["Requires grad"] = "c"
    table.align["Trainable parameters"] = "r"

    total_params = 0
    for name, parameter in model.named_parameters():
        
        requires_grad = True
        params = parameter.numel()
        
        if not parameter.requires_grad:
            requires_grad = False
            params = 0
        
        table.add_row([name, requires_grad, params])
        total_params += params
    
    print(table)
    print(f"Total Trainable Params: {total_params:,}")
    return total_params


def seed_everything(seed=0, harsh=True):
    """
    Seeds all random functions for reproducibility
    -------------------------------------
    Args:
        seed (int, optional): seed value. Defaults to 0.
        harsh (bool, optional): torch backend deterministic. Defaults to False.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
    if harsh:
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
