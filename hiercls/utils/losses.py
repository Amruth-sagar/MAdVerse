import torch
import torch.nn as nn
from torch.nn.functional import cross_entropy, softmax, log_softmax
from hiercls.datasets.data_utils import *
from hiercls.utils.registry import registry
import ipdb
from nltk import Tree
from copy import deepcopy
from math import exp, fsum, log2
from copy import deepcopy
#=========================================================================================#
#                                   LEAF_LEVEL_ONLY LOSSES                                #
#=========================================================================================#

def map_nodes_to_integers(hier_info):
    """
    This function will map integers given to class names, to string labels
    -------------------------------------------------------
    Args:
        class_tree: an nltk.Tree object, which is the tree representation of the 
                    class hierarchy in the dataset.
    """

    nodes = [(x[-1], str(len(x)) + '_' + str(x[-1])) for x in hier_info.hierarchy_int]

    return (dict([(node, index) for (index, node) in nodes]),
            dict([(index, node) for (index, node) in nodes]))



@registry.add_loss_to_registry("simple_ce_l")
class SimpleCE(nn.Module):
    def __init__(self, loss_config, hier_info):
        super().__init__()
        self.node_to_int, self.int_to_node = map_nodes_to_integers(hier_info)
        self.leaf_only = True

    def forward(self, logits, level_wise_targets):

        device = logits[-1].device
        leaf_level_targets = torch.tensor([hier_annot[-1] for hier_annot in level_wise_targets]).to(device)

        loss = cross_entropy(
            logits[-1],
            leaf_level_targets,
        )
        return loss

@registry.add_loss_to_registry("barz_denzler_l")
class BarzAndDenzlerLoss(nn.Module):
    def __init__(self, loss_config, hier_info):
        super().__init__()
        self.num_levels = hier_info.num_levels
        self.num_cls_each_level = hier_info.num_cls_in_level
        self.node_to_int, self.int_to_node = map_nodes_to_integers(hier_info)
        self.leaf_only = True

        self.level_wise_bd_embeddings = []
        self.level_wise_lca_heights = get_lca_height_of_leaves(
            hier_info.hierarchy_int, hier_info.dataset_config
        )
        self.cost_matrices = construct_cost_matrix(
            hier_info.hierarchy_int,
            self.level_wise_lca_heights,
            hier_info.num_cls_in_level,
            only_last_level=True,
            leaves_mapping=self.node_to_int
        )

        # In the paper, distance is LCA(node1, node2) / height_of_tree
        # The similarity is given by (1 - distance)
        self.node_similarity_all_levels = [
            1 - (lca_heights / self.num_levels)
            for lca_heights in self.cost_matrices
        ]

        for level in range(len(self.node_similarity_all_levels)):
            current_level_bd_embeddings = get_unitsphere_embeddings(
                self.node_similarity_all_levels[level]
            )
            self.level_wise_bd_embeddings.append(current_level_bd_embeddings)

    def forward(self, logits, level_wise_targets):
        """
        logits:             logits produced for the levels of hierarchy by the model. ( list of [batch, <num_cls>])
        level_wise_target:  Target class for that particular level. (tensor of [batch, num_levels])
        """
        device = logits[-1].device

        target_indices = [hier_annot[-1].item() for hier_annot in level_wise_targets]

        leaf_bd_embeddings = self.level_wise_bd_embeddings[-1][target_indices, :]

        gt_bd_embeddings = leaf_bd_embeddings.to(device)
        network_bd_embeddings = logits[-1]

        # Since the target embeddings are in unit-hypersphere in n-dimensions
        # L2 normalization is applied to the network's output.
        network_bd_embeddings = nn.functional.normalize(
            network_bd_embeddings, p=2, dim=1
        )

        # Here, we measure the similarity between images and the bd_embeddings.
        inner_products = torch.sum(gt_bd_embeddings * network_bd_embeddings, dim=1)
        correlation_loss = torch.mean(1 - inner_products)

        return correlation_loss


"""
label-embedding approach from "Making better mistakes": https://arxiv.org/pdf/1912.09393.pdf
"""
@registry.add_loss_to_registry("soft_labels_l")
class SoftLabelsLoss(nn.Module):
    def __init__(self, loss_config, hier_info):
        super().__init__()
        self.num_levels = hier_info.num_levels
        self.node_to_int, self.int_to_node = map_nodes_to_integers(hier_info)
        self.leaf_only = True

        self.beta_value = loss_config.beta_value
        level_wise_lca_heights = get_lca_height_of_leaves(
             hier_info.hierarchy_int, hier_info.dataset_config
        )
        last_level_lca_heights = construct_cost_matrix(
            
            hier_info.hierarchy_int,
            level_wise_lca_heights,
            hier_info.num_cls_in_level,
            only_last_level=True,
            leaves_mapping=self.node_to_int
        )

        # In paper, the distance is LCA(node1, node2) / height_of_tree
        self.soft_labels = (
            -1 * self.beta_value * (last_level_lca_heights[0] / self.num_levels)
        )

        # Either consider a row as an embedding, or a column. This is
        # important as we are applying the softmax along that dim.
        # we consider a row as an embedding of a class.
        self.soft_labels = softmax(self.soft_labels, dim=1)

    def forward(self, logits, level_wise_targets):
        """
        logits:             logits for the leaf level classes by the model. ( list of [batch, <num_>])
        level_wise_target:  Target class for that particular level. (tensor of [batch, num_levels])
        """

        device = logits[0].device

        leaf_level_probs = log_softmax(logits[-1], dim=1)

        leaf_target_indices = [hier_annot[-1].item() for hier_annot in level_wise_targets]

        gt_soft_labels = self.soft_labels[leaf_target_indices, :]

        soft_loss = -1 * torch.sum(
            torch.mul(leaf_level_probs, gt_soft_labels.to(device)), dim=1
        )

        # Reduction = mean
        soft_loss = torch.mean(soft_loss)

        return soft_loss

@registry.add_loss_to_registry("dot_l")
class DiscreteOptimalTransportLoss(nn.Module):  
    def __init__(self, loss_config, hier_info):
        super().__init__()
        self.num_levels = hier_info.num_levels
        self.node_to_int, self.int_to_node = map_nodes_to_integers(hier_info)
        self.leaf_only = True

        self.pth_power = loss_config.pth_power
        
        level_wise_tie_distances = get_tie_of_leaves(
             hier_info.hierarchy_int, hier_info.dataset_config
        )
        self.level_wise_tie_distances = construct_cost_matrix(
            hier_info.hierarchy_int,
            level_wise_tie_distances,
            hier_info.num_cls_in_level,
            only_last_level=True,
            leaves_mapping=self.node_to_int 
        ) 

    def forward(self, logits, level_wise_targets):
        """
        logits:             logits for leaf level classes by the model. ( list of [batch, <num_leaves>])
        level_wise_target:  Target class for that particular level. (tensor of [batch, num_levels])
        """
        device = logits[0].device

        softmax_leaf_level = softmax(logits[-1], dim=1)

        target_indices = [hier_annot[-1].item() for hier_annot in level_wise_targets]

        # tie_source_target (batch, num_leaves) : if a samples target is 'a', then its row
        # contains all the tie between 'a' and all other leaf nodes.
        tie_source_target = self.level_wise_tie_distances[-1][target_indices, :]
        tie_source_target_transformed = torch.pow(tie_source_target, self.pth_power)

        dot_loss = torch.sum(
                torch.mul(
                    tie_source_target_transformed.to(device),
                    softmax_leaf_level,
                ),
                dim=1,
            )
        
        # Reduction = Mean
        # NOTE: since this is only single leveled, there is no need of weighting, 
        # which is done in the paper as they were applying DOT at every level.
        dot_loss = torch.mean(dot_loss)

        return dot_loss

"""
Code taken from "Making better mistakes" : https://github.com/fiveai/making-better-mistakes/blob/master/better_mistakes/model/losses.py
minor modifications has been done.
"""
@registry.add_loss_to_registry("hxe_l")
class HXELoss(nn.Module):
    def __init__(self, loss_config, hier_info):
        super().__init__()
        self.num_levels = hier_info.num_levels

        value = loss_config.alpha_value
        hierarchy_tree = hier_info.hierarchy_tree

        leaf_to_int_map = list(
                zip(
                    [str(len(x))+"_"+str(x[-1]) for x in hier_info.hierarchy_int],
                    list(range(len(hier_info.hierarchy_int))),
                )
            )
        
        self.node_to_int = dict(leaf_to_int_map)
        self.int_to_node = dict([(value, key) for key, value in self.node_to_int.items()])
        self.leaf_only = True

        
        leaf_classes_list = [pair[0] for pair in leaf_to_int_map]

        weights_tree = self.get_exponential_weighting(
            hierarchy_tree, value, loss_config.normalize
        )
        assert hierarchy_tree.treepositions() == weights_tree.treepositions()

        # Ex: if tree position is (0,1,0,2), it is the positions of children node on the path.
        # 0th child of root --> 1st child of previous node --> 0th child of previous node --> 2nd node L3 of previous node
        positions_leaves = {
            self.get_label(hierarchy_tree[p]): p
            for p in hierarchy_tree.treepositions("leaves")
        }
        num_classes = len(positions_leaves)

        # we use classes in the given order
        positions_leaves = [positions_leaves[c] for c in leaf_classes_list]

        # the tree positions of all the edges (we use the bottom node position)
        positions_edges = hierarchy_tree.treepositions()[
            1:
        ]  # the first one is the origin

        # map from position tuples to leaf/edge indices
        index_map_leaves = {
            positions_leaves[i]: i for i in range(len(positions_leaves))
        }
        index_map_edges = {positions_edges[i]: i for i in range(len(positions_edges))}

        # edge indices corresponding to the path from each index to the root
        edges_from_leaf = [
            [index_map_edges[position[:i]] for i in range(len(position), 0, -1)]
            for position in positions_leaves
        ]

        # get max size for the number of edges to the root
        num_edges = max([len(p) for p in edges_from_leaf])

        def get_leaf_positions(position):
            node = hierarchy_tree[position]
            if isinstance(node, Tree):
                return node.treepositions("leaves")
            else:
                return [()]

        # indices of all leaf nodes for each edge index
        leaf_indices = [
            [index_map_leaves[position + leaf] for leaf in get_leaf_positions(position)]
            for position in positions_edges
        ]

        # save all relevant information as pytorch tensors for computing the loss on the gpu
        onehot_den = torch.zeros([num_classes, num_classes, num_edges])
        onehot_num = torch.zeros([num_classes, num_classes, num_edges])
        weights = torch.zeros([num_classes, num_edges])

        self.register_buffer("onehot_den", onehot_den, persistent=False)
        self.register_buffer("onehot_num", onehot_num, persistent=False)
        self.register_buffer("weights", weights, persistent=False)

        # one hot encoding of the numerators and denominators and store weights
        for i in range(num_classes):
            for j, k in enumerate(edges_from_leaf[i]):
                self.onehot_num[i, leaf_indices[k], j] = 1.0
                self.weights[i, j] = self.get_label(weights_tree[positions_edges[k]])
            for j, k in enumerate(edges_from_leaf[i][1:]):
                self.onehot_den[i, leaf_indices[k], j] = 1.0
            self.onehot_den[
                i, :, j + 1
            ] = 1.0  # the last denominator is the sum of all leaves

    def get_label(self, node):
        if isinstance(node, Tree):
            return node.label()
        else:
            return node

    def get_exponential_weighting(self, hierarchy_tree, value, normalize=True):
        weights_tree = deepcopy(hierarchy_tree)
        all_weights = []
        for position in weights_tree.treepositions():
            current_node = weights_tree[position]
            weight = exp(-value * len(position))
            all_weights.append(weight)

            if isinstance(current_node, Tree):
                current_node.set_label(weight)
            else:
                weights_tree[position] = weight

        total = fsum(all_weights)

        if normalize:
            for position in weights_tree.treepositions():
                current_node = weights_tree[position]
                if isinstance(current_node, Tree):
                    current_node.set_label(current_node.label() / total)
                else:
                    weights_tree[position] /= total

        return weights_tree

    def forward(self, logits, level_wise_target):
        """
        Foward pass, computing the loss.

        Args:
            logits :            list of logits. As HXE applies only on leaf level, it is a list of
                                one element, which is in the shape of [batch, num_cls_in_leaf_level]
            level_wise_target:  Target class for that particular level. (tensor of [batch, num_levels])
        """
        # The shape becomes [batch, 1, num_cls_in_leaf_level]
        inputs = softmax(logits[-1], dim=1)
        inputs = torch.unsqueeze(inputs, 1)

        target_indices = [hier_annot[-1].item() for hier_annot in level_wise_target]

        num = torch.squeeze(
            torch.bmm(inputs, self.onehot_num[target_indices])
        )
        den = torch.squeeze(
            torch.bmm(inputs, self.onehot_den[target_indices])
        )
            
        idx = num != 0
        num[idx] = -torch.log(num[idx] / den[idx])
        num = torch.sum(
            torch.flip(self.weights[target_indices] * num, dims=[1]), dim=1
        )

        return torch.mean(num)


#=========================================================================================#
#                               MULTI-LEVEL LOSSES
#=========================================================================================#


@registry.add_loss_to_registry("sum_ce")
class SumCELoss(nn.Module):
    def __init__(self, loss_config, hier_info):
        super().__init__()
        self.num_levels = hier_info.num_levels
        self.leaf_only = False

    def forward(self, logits_from_all_levels, level_wise_target):
        """
        logits_from_all_levels: logits produced for each level by the model. ( list of [batch, <num_cls>])
        level_wise_target:      Target class for that particular level. (tensor of [batch, num_levels])
        """
        cross_entropy_each_level = []
        for current_level in range(self.num_levels):
            cross_entropy_each_level.append(
                cross_entropy(
                    logits_from_all_levels[current_level],
                    level_wise_target[:, current_level],
                )
            )

        return sum(cross_entropy_each_level)


"""
In the paper, discrete optimal transport is applied levelwise, and a weighted summation of losses from each 
level is taken (weights of a loss depends on the height of that level). The below implementation is level-wise. 
"""
@registry.add_loss_to_registry("dot")
class DiscreteOptimalTransportLossML(nn.Module):  
    def __init__(self, loss_config, hier_info):
        super().__init__()
        self.num_levels = hier_info.num_levels
        self.leaf_only = False

        self.pth_power = loss_config.pth_power
        level_wise_tie_distances = get_level_wise_tie_distances(
             hier_info.hierarchy_int, hier_info.dataset_config
        )
        self.level_wise_tie_distances = construct_cost_matrix(
            hier_info.hierarchy_int,
            level_wise_tie_distances,
            hier_info.num_cls_in_level,
            only_last_level=False,
        )

    def forward(self, logits_from_all_levels, level_wise_target):
        """
        logits_from_all_levels: logits produced for levels in hierarchy by the model. ( list of [batch, <num_cls>])
        level_wise_target:      Target class for that particular level. (tensor of [batch, num_levels])
        """
        device = logits_from_all_levels[0].device

        softmax_all_levels = []

        for current_level in range(len(logits_from_all_levels)):
            softmax_all_levels.append(
                softmax(logits_from_all_levels[current_level], dim=1)
            )

        dot_loss_each_level = []
        for current_level in range(len(logits_from_all_levels)):
            # The shape of tie_source_target will be [batch, num_cls_in_that_level]
            target_indices = [x[current_level] for x in level_wise_target if len(x) >= current_level+1]
            tie_source_target = self.level_wise_tie_distances[current_level][
                target_indices, :
            ]

            # In paper, they mentioned f(dij), where dij is TIE between i and j
            # and the f is a function which is pth power of this distance.
            tie_source_target_transformed = torch.pow(tie_source_target, self.pth_power)

            # Summing is done into two parts, for readability sake
            dot_loss_all_samples = torch.sum(
                torch.mul(
                    tie_source_target_transformed.to(device),
                    softmax_all_levels[current_level],
                ),
                dim=1,
            )

            # Reduction = mean
            dot_loss_batch = torch.mean(dot_loss_all_samples)

            dot_loss_each_level.append(dot_loss_batch)

        # MODIFICATION: weighting the level wise losses with log(level+1)
        # NOTE: here level is one indexed, 
        # as weighting mentioned in paper does not handle edge cases.
        level_wise_weights = torch.tensor(
            [log2((level + 1)) for level in range(len(logits_from_all_levels), 0, -1)]
        )

        weighted_dot_loss_each_level = sum(
            [
                weight * dot_loss_each_level[level]
                for level, weight in enumerate(level_wise_weights)
            ]
        )

        return weighted_dot_loss_each_level