from enum import Enum
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    top_k_accuracy_score,
)
from collections import defaultdict
from numpy import asarray
import numpy as np
import ipdb


non_hier_metrics = {
    "accuracy": accuracy_score,
    "f1_score": lambda y_true, y_pred: f1_score(
        y_true=y_true, y_pred=y_pred, average="macro", zero_division=0
    ),
    "precision": lambda y_true, y_pred: precision_score(
        y_true=y_true, y_pred=y_pred, average="macro", zero_division=0
    ),
    "recall": lambda y_true, y_pred: recall_score(
        y_true=y_true, y_pred=y_pred, average="macro", zero_division=0
    ),
}

non_hier_metrics_top_k = {
    "acc_top_2": lambda y_true, y_scores: top_k_accuracy_score(
        y_true=y_true, y_score=y_scores, k=2
    ),
    "acc_top_5": lambda y_true, y_scores: top_k_accuracy_score(
        y_true=y_true, y_score=y_scores, k=5
    ),
    "acc_top_10": lambda y_true, y_scores: top_k_accuracy_score(
        y_true=y_true, y_score=y_scores, k=10
    ),
}




def calculate_metrics(metrics, pred_labels, gt_labels, num_levels, probabilities, loss_class=None, criterion=None, bet_on_leaf = None):
    """
    NOTE: criterion is being sent, for the case where, for a leaf only method we use bet_on_leaf_level, which will remap 
    the classes to their original mapping, but to calculate top-k, we need probabilities, but they are given in older 
    mapping. Hence to remap, we am using the criterion again to convert back to the mapping that the loss class gives. 
    """
    
    print("Number of gt labels per sample: ", len(gt_labels[0]))
    print("Number of pred labels per sample: ", len(pred_labels[0]))
    
    # Since the loss classes mentioned in the list do not output probabilities, hence top_k cannot be 
    # calculated for these loss classes.
    
    if loss_class in ['barz_denzler_l']:
        # There is only one level as 'barz_denzler_l' is a leaf level class. 
        # If we have enabled bet_on_leaf_level, we would have predictions of all levels
        if len(pred_labels[0]) == 1:
            dict_1 = calculate_metrics_last_level(metrics, pred_labels, gt_labels, num_levels)
        else:
            dict_1 = calculate_metrics_all_levels(metrics, pred_labels, gt_labels, num_levels)
        
        return {**dict_1}


    else:

        if len(pred_labels[0]) == 1 and len(probabilities) == 1:
            # There is only one level. 
            dict_1 = calculate_metrics_last_level(metrics, pred_labels, gt_labels, num_levels)
            dict_2 = calculate_metrics_last_level_topk(metrics, gt_labels, num_levels, probabilities)

        elif len(probabilities) == 1 and bet_on_leaf:
            # This is the case where 'bet_on_leaf_level' is TRUE, which is enabled to get
            # parent level predictions by traversing from predicted leaf node to root (top-most level)

            # In this case, we have #level number of predicted labels, but only leaf-level's probs.
            dict_1 = calculate_metrics_all_levels(metrics, pred_labels, gt_labels, num_levels)

            dict_2 = calculate_metrics_last_level_topk(metrics, gt_labels, num_levels, probabilities)
        
        else:
            # We have one classifier per level in the hierarchy
            dict_1 = calculate_metrics_all_levels(metrics, pred_labels, gt_labels, num_levels)
            dict_2 = calculate_metrics_all_levels_topk(metrics, gt_labels, num_levels, probabilities)

        return {**dict_1, **dict_2}




def calculate_metrics_last_level(metrics, last_level_pred_labels, gt_hier_labels, num_levels):
    """
    This function will be called when only the leaf level is required for the loss calculation,
    and unlike classification heads for each level, there is only one present in this case.
    ----------------------------------------------------------------------------------------------------------------
    Args:
        metrics: list of strings, which tells what metrics to calculate
        last_level_pred_labels: this is in the form [[pred1],[pred2],[pred3], ...]
        gt_hier_labels: For each sample, it has ground truth for each level of 
                        hierarchy  [[a1, b1, c1 ..],
                                    [a2, b2, c2 ..],
                                    ...]
        last_level_probs: this is a List[Tensor] which contains a single tensor of shape [num_samples, num_clas_in_last_level]
    """
    scores = defaultdict(list)

    for m in metrics:
        if m not in non_hier_metrics and m not in non_hier_metrics_top_k:
            raise ValueError(
                "The given metric {0} is not the correct name. Retry by giving the correct name".format(
                    m
                )
            )
        if m in non_hier_metrics:
            # As we only have predictions for the leaf level, we directly save it under the level number of leaves, 
            # which is same as num_levels.
            scores[m] = [
                (
                    "level_{}".format(num_levels),
                    non_hier_metrics[m](
                        y_true=[label[-1] for label in gt_hier_labels],
                        y_pred=[label[-1] for label in last_level_pred_labels],
                    ),
                )
            ]
    return scores


def calculate_metrics_last_level_topk(metrics, gt_hier_labels, num_levels, last_level_probs):
    """
    This function will be called when only the leaf level is required for the loss calculation,
    and unlike classification heads for each level, there is only one present in this case.
    ----------------------------------------------------------------------------------------------------------------
    Args:
        metrics: list of strings, which tells what metrics to calculate
        last_level_pred_labels: this is in the form [[pred1],[pred2],[pred3], ...]
        gt_hier_labels: For each sample, it has ground truth for each level of 
                        hierarchy  [[a1, b1, c1 ..],
                                    [a2, b2, c2 ..],
                                    ...]
        last_level_probs: this is a List[Tensor] which contains a single tensor of shape [num_samples, num_clas_in_last_level]
    """
    scores = defaultdict(list)

    for m in metrics:
        if m not in non_hier_metrics and m not in non_hier_metrics_top_k:
            raise ValueError(
                "The given metric {0} is not the correct name. Retry by giving the correct name".format(
                    m
                )
            )
        if m in non_hier_metrics_top_k:
            scores[m] = [
                (
                    "level_{}".format(num_levels),
                    non_hier_metrics_top_k[m](
                        y_true=[label[-1] for label in gt_hier_labels],
                        y_scores=last_level_probs[0].detach().cpu().numpy(),
                    ),
                )
            ]
    return scores


def calculate_metrics_all_levels(metrics, pred_hier_labels, gt_hier_labels, num_levels):
    """
    This function calculates each metrics, for all the levels in the hierarchy.
    ----------------------------------------------------------------------------------------------------------------
    Args:
        metrics: list of strings, which tells what metrics to calculate
        pre_hier_labels: For each sample, there are num_levels of predictions
                                             [[p11, p12, p13 ..],
                                              [p21, p22, p23 ..], 
                                              ...]
        gt_hier_labels: For each sample, it has ground truth for each level of 
                        hierarchy  [[a1, b1, c1 ..],
                                    [a2, b2, c2 ..],
                                    ...]
        y_scores_all_levels: this is a List[Tensor, Tensor, ...] which contains tensors of shape [num_samples, num_clas_in_that_level]
    """
    scores = defaultdict(list)

    for m in metrics:
        if m not in non_hier_metrics and m not in non_hier_metrics_top_k:
            raise ValueError(
                "The given metric {0} is not the correct name. Retry by giving the correct name".format(
                    m
                )
            )
        if m in non_hier_metrics:
            scores[m] = [
                (
                    "level_{}".format(level + 1),
                    non_hier_metrics[m](
                        y_true=[label[level] for label in gt_hier_labels],
                        y_pred=[label[level] for label in pred_hier_labels],
                    ),
                )
                for level in range(num_levels)
            ]
    return scores


def calculate_metrics_all_levels_topk(metrics, gt_hier_labels, num_levels, y_scores_all_levels):
    """
    This function calculates each metrics, for all the levels in the hierarchy.
    ----------------------------------------------------------------------------------------------------------------
    Args:
        metrics: list of strings, which tells what metrics to calculate
        pre_hier_labels: For each sample, there are num_levels of predictions
                                             [[p11, p12, p13 ..],
                                              [p21, p22, p23 ..], 
                                              ...]
        gt_hier_labels: For each sample, it has ground truth for each level of 
                        hierarchy  [[a1, b1, c1 ..],
                                    [a2, b2, c2 ..],
                                    ...]
        y_scores_all_levels: this is a List[Tensor, Tensor, ...] which contains tensors of shape [num_samples, num_clas_in_that_level]
    """
    scores = defaultdict(list)

    for m in metrics:
        if m not in non_hier_metrics and m not in non_hier_metrics_top_k:
            raise ValueError(
                "The given metric {0} is not the correct name. Retry by giving the correct name".format(
                    m
                )
            )
        if m in non_hier_metrics_top_k:
            scores[m] = [
                (
                    "level_{}".format(level + 1),
                    non_hier_metrics_top_k[m](
                        y_true=[label[level] for label in gt_hier_labels],
                        y_scores=y_scores_all_levels[level].detach().cpu().numpy(),
                    ),
                )
                for level in range(num_levels)
            ]
    return scores



# NOTE: Here, lw in the method name means 'level wise'

def lca_height_lw(pred_labels, gt_labels, tree_stats):
    height_of_lca = get_height_of_LCA(pred_labels, gt_labels, tree_stats)
    return np.mean(height_of_lca, axis=0)


def lca_height_mistakes_lw(pred_labels, gt_labels, tree_stats):
    height_of_lca = get_height_of_LCA(pred_labels, gt_labels, tree_stats)
    return height_of_lca.sum(0) / (height_of_lca != 0).sum(0)


def tie_lw(pred_labels, gt_labels, tree_stats):
    tree_induced_loss = get_tree_induced_loss(pred_labels, gt_labels, tree_stats)
    return np.mean(tree_induced_loss, axis=0)


def tie_mistakes_lw(pred_labels, gt_labels, tree_stats):
    tree_induced_loss = get_tree_induced_loss(pred_labels, gt_labels, tree_stats)
    return tree_induced_loss.sum(0) / (tree_induced_loss != 0).sum(0)


hier_metrics = {
    "lca_height_lw": lca_height_lw,
    "lca_height_mistakes_lw": lca_height_mistakes_lw,
    "tie_lw": tie_lw,
    "tie_mistakes_lw": tie_mistakes_lw,
}




def calculate_hier_metrics(metrics, pred_labels, gt_labels, tree_stats, num_levels, loss_class):
    if len(pred_labels[0]) == 1:
        return calculate_hier_metrics_last_level(metrics, pred_labels, gt_labels, tree_stats, num_levels)
    else:
        return calculate_hier_metrics_each_level(metrics, pred_labels, gt_labels, tree_stats, num_levels)


def calculate_hier_metrics_last_level(metrics, pred_labels, gt_labels, tree_stats, num_levels):
    scores = defaultdict(list)

    for m in metrics:
        if m not in hier_metrics:
            raise ValueError(
                "The given metric {0} is not the correct/implemented. Retry by giving the correct name".format(
                    m
                )
            )

        result = hier_metrics[m](pred_labels, gt_labels, tree_stats)
        scores[m] = [
            ("level_{}".format(num_levels), result[0])
        ]
    return scores

def calculate_hier_metrics_each_level(metrics, pred_labels, gt_labels, tree_stats, num_levels):
    scores = defaultdict(list)

    for m in metrics:
        if m not in hier_metrics:
            raise ValueError(
                "The given metric {0} is not the correct/implemented. Retry by giving the correct name".format(
                    m
                )
            )

        result = hier_metrics[m](pred_labels, gt_labels, tree_stats)
        scores[m] = [
            ("level_{}".format(level + 1), result[level]) for level in range(num_levels)
        ]
    return scores


def get_height_of_LCA(pred_labels, gt_labels, tree_stats):
    """
    This function returns the lca distances between pred and gt
    ------------------------------------------------------------------
    Args:
        pred_labels:    a list, whose length is batch size, and elements 
                        in the list vary in length, almost around num_levels

        gt_labels:      similar to pred_labels, but instead has ground truths

        tree_stats:     contains 1) lca height of leaf level or all levels,
                                 2) tie_distance of leaf or all levels,
                                 3) int_to_node mapping, in case of leaf level 

    """
    height_of_lca = []
    
    lca_heights = tree_stats.lca_heights


    if len(pred_labels[0]) == 1:

        # When we are dealing with LEAF LEVEL ONLY, LCA_HEIGHT HAS ('level_num1', 'level_num2'): lca_height
        # When we are doing multilevel (or leaf only but all the leaves at same node) LCA_HEIGHTS has (num_1, num_2): lca_height
        
        if tree_stats.int_to_node:
            print('Mapping for integer to node names found. Using it to calculate TIE by converting integers to names')
            int_to_node = tree_stats.int_to_node

            # Convert pred_labels and gt_labels back to node labels
            gt_labels = [[int_to_node[label] for label in target] for target in gt_labels]
            pred_labels = [[int_to_node[label] for label in predicted] for predicted in pred_labels]
        else:
            print("No int_to_node mapping found.")
        
        for idx in range(len(pred_labels)):
            pred = pred_labels[idx]
            gt = gt_labels[idx]

            # Each sample's prediction comes in a list, which is done so
            # keeping in mind that for multilevel case, it has num_levels 
            # number of labels in it.
            node_1 = pred[-1]
            node_2 = gt[-1]

            if node_1 == node_2:
                lca_height = 0
            elif (node_1, node_2) in lca_heights[-1]:
                lca_height = lca_heights[-1][(node_1, node_2)]
            else:
                lca_height = lca_heights[-1][(node_2, node_1)]

            height_of_lca.append([lca_height])

        height_of_lca = asarray(height_of_lca)
    
    else:
        
        for idx in range(len(pred_labels)):
            pred = pred_labels[idx]
            gt = gt_labels[idx]

            pred_vs_gt_lca = []
            for curr_level, (node_1, node_2) in enumerate(list(zip(pred, gt))):
                if node_1 == node_2:
                    pred_vs_gt_lca.append(0)
                    continue
                if (node_1, node_2) in lca_heights[curr_level]:
                    pred_vs_gt_lca.append(lca_heights[curr_level][(node_1, node_2)])
                else:
                    pred_vs_gt_lca.append(lca_heights[curr_level][(node_2, node_1)])

            height_of_lca.append(pred_vs_gt_lca)

        height_of_lca = asarray(height_of_lca)

    return height_of_lca


def get_tree_induced_loss(
    pred_labels, gt_labels, tree_stats
):
    """
    This function returns the tie distances between pred and gt
    ------------------------------------------------------------------
    Args:
        pred_labels:    a list, whose length is batch size, and elements 
                        in the list vary in length, almost around num_levels

        gt_labels:      similar to pred_labels, but instead has ground truths

        tree_stats:     contains 1) lca height of leaf level or all levels,
                                 2) tie_distance of leaf or all levels,
                                 3) int_to_node mapping, in case of leaf level 
    """

    tree_induced_loss = []
    tie_distances = tree_stats.tie_distances

    num_levels = max([len(x) for x in gt_labels])
    
    if len(pred_labels[0]) == 1:

        # When we are dealing with LEAF LEVEL ONLY, LCA_HEIGHT HAS ('level_num1', 'level_num2'): lca_height
        # When we are doing multilevel (or leaf only but all the leaves at same node) LCA_HEIGHTS has (num_1, num_2): lca_height


        if tree_stats.int_to_node:
            print('Mapping for integer to node names found. Using it to calculate LCA by converting integers to names')
            int_to_node = tree_stats.int_to_node

            # Convert pred_labels and gt_labels back to node labels
            gt_labels = [[int_to_node[label] for label in target] for target in gt_labels]
            pred_labels = [[int_to_node[label] for label in predicted] for predicted in pred_labels]
        else:
            print("No int_to_node mapping found.")
        
        for idx in range(len(pred_labels)):
            pred = pred_labels[idx]
            gt = gt_labels[idx]

            node_1 = pred[-1]
            node_2 = gt[-1]

            if node_1 == node_2:
                tie_dist = 0
            else:
                if (node_1, node_2) in tie_distances[-1]:
                    tie_dist = tie_distances[-1][(node_1, node_2)]
                else:
                    tie_dist = tie_distances[-1][(node_2, node_1)]
            
            tree_induced_loss.append([tie_dist])

        tree_induced_loss = asarray(tree_induced_loss)

    else:

        for idx in range(len(pred_labels)):
            pred = pred_labels[idx]
            gt = gt_labels[idx]

            pred_gt_tie = []
            for curr_level, (node_1, node_2) in enumerate(list(zip(pred, gt))):

                if node_1 == node_2:
                    pred_gt_tie.append(0)
                    continue
                if (node_1, node_2) in tie_distances[curr_level]:
                    tie_dist = tie_distances[curr_level][(node_1, node_2)]
                else:
                    tie_dist = tie_distances[curr_level][(node_2, node_1)]

                pred_gt_tie.append(tie_dist)

            tree_induced_loss.append(pred_gt_tie)

        tree_induced_loss = asarray(tree_induced_loss)

    return tree_induced_loss
