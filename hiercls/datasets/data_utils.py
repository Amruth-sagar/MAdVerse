import numpy as np
from collections import defaultdict
from itertools import combinations
import pickle
import torch
import pathlib
from nltk import Tree
import ipdb
from tqdm import tqdm
import numpy as np

# The below class DotDict, will enable us to access the elements 
# in the dictionary using the dot notation.
class DotDict:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            self.__dict__[key] = value

    def __getattr__(self, key):
        if key in self.__dict__:
            return self.__dict__[key]
        else: 
            return None


def get_stats_of_hierarchy(hierarchy_int):
    """
    Get statistics like number of levels and number of classes in each level of hierarchy,
    either balanced or unbalanced.
    -------------------------------------------------------------------------------------
    Args:
        hierarchy_int: integer mapping of classes in each, in the form [[1,2,3],
                                                                        [1,3,15]...]
    """
    num_levels = max([len(x) for x in hierarchy_int])

    num_cls_in_each_level = []

    for level in range(num_levels):
        nodes_in_curr_level = []
        
        for node_labels in hierarchy_int:
            try:
                nodes_in_curr_level.append(node_labels[level])
            except:
                continue
        num_cls_in_each_level.append(len(set(nodes_in_curr_level)))

    return num_levels, num_cls_in_each_level

def get_level_wise_tie_distances(hierarchy_int, dataset_config):
    
    # the function returns a dict per level, with entries in each as (node1, node2) = height
    level_wise_lca_heights = get_level_wise_lca_heights(hierarchy_int, dataset_config)
    
    # Since the LCA height are calculated between the nodes in the same level,
    # TIE (tree induced loss) will be simplified to 2 * LCA
    level_wise_tie = [
        dict(zip(list(lca_height.keys()), [height*2 for height in list(lca_height.values())])) 
        for lca_height in level_wise_lca_heights
    ]

    return level_wise_tie

def get_level_wise_lca_heights(hierarchy_int, dataset_config):
    """
    This function returns the HEIGHT OF LCA, calculated between the nodes of same level.
    For example, this function does not calculate the lca distance between level-2 node
    and level-3 node.

    Note: for the top-most level, the lca distances between any pair of nodes
    except (i,i) is same, assuming that there is a dummy root node.
    -----------------------------------------------------------------------------------
    Args:
        hierarchy_int: integer mapping of classes in each, in the form [[1,2,3,4],
                                                                        [1,3,15]...]
        dataset_config: dataset_config
    """

    distances_file = pathlib.Path(dataset_config.annot_dir) / pathlib.Path(
        dataset_config.distances_file
    )
    
    # If exists, read the saved distances files
    # and return it.
    if distances_file.is_file():
        with open(distances_file, "rb") as infile:
            distances = pickle.load(infile)
        return distances
    

    root_to_leaf_paths_levelwise = []
    num_levels = len(hierarchy_int[0])

    # Here we treat -1 as the index of dummy root node.

    for level in range(num_levels):
        temp = list(
            set([ tuple([-1]) +  tuple(hier_label[: level + 1]) for hier_label in hierarchy_int])
        )
        root_to_leaf_paths_levelwise.append(temp)

    height_lca_levelwise = []

    for level in range(num_levels):
        curr_level_height_lca = defaultdict(tuple)

        num_classes_curr_level = len(
            set([hier_label[level] for hier_label in hierarchy_int])
        )
        nC2_pairs = list(combinations(list(range(num_classes_curr_level)), 2))

        for i, j in nC2_pairs:
            path_1 = root_to_leaf_paths_levelwise[level][i]
            path_2 = root_to_leaf_paths_levelwise[level][j]

            similar_until = 0
            for pos in range(len(path_1)):
                if path_1[pos] == path_2[pos]:
                    continue
                else:
                    similar_until = pos
                    break

            node_1 = path_1[-1]
            node_2 = path_2[-1]
            curr_level_height_lca[(node_1, node_2)] = len(path_1) - similar_until

        height_lca_levelwise.append(curr_level_height_lca)

    # Saving the calculated distances.
    with open(distances_file, "wb") as outfile:
        pickle.dump(height_lca_levelwise, outfile)

    return height_lca_levelwise


def lca_distance(tree, leaf_index_1, leaf_index_2):
    """
    This function returns the distance of LCA, which is also
    referred as TIE(tree induced error) by few papers. It is the
    shortest path from one node to other node, which always goes
    throught LCA(node1, node2)
    -----------------------------------------------------------
    Args:
        tree: nltk Tree object of the class hierarchy
        leaf_index_1: leaf index
        laef_index_2: leaf index
    """
    if leaf_index_1 == leaf_index_2:
        return 0

    path1 = tree.leaf_treeposition(leaf_index_1)
    path2 = tree.leaf_treeposition(leaf_index_2)

    num_nodes_similar = 0
    for i in range(min(len(path1), len(path2))):
        if path1[i] != path2[i]:
            break
        num_nodes_similar = i+1

    return len(path1) + len(path2) - (2 * num_nodes_similar)


def lca_height(tree, leaf_index_1, leaf_index_2):
    """
    This function returns the height of LCA, by subtracting
    the tree height with the lenght of common path between two
    nodes
    -----------------------------------------------------------
    Args:
        tree: nltk Tree object of the class hierarchy
        leaf_index_1: leaf index
        laef_index_2: leaf index
    """
    if leaf_index_1 == leaf_index_2:
        return 0

    path1 = tree.leaf_treeposition(leaf_index_1)
    path2 = tree.leaf_treeposition(leaf_index_2)

    # Subtracting 1 since nltk.height() tells the number of levels in tree (root node is level-1).
    tree_height = tree.height() - 1

    # Default height is the height of tree
    height = tree_height

    for i in range(min(len(path1), len(path2))):
        if path1[i] != path2[i]:
            height = tree_height - i
            break
    return height

def get_tie_of_leaves(hierarchy_int, dataset_config):
    """
    This function returns the HEIGHT OF LCA, calculated between the leaf nodes.

    -----------------------------------------------------------------------------------
    Args:
        hierarchy_int: integer mapping of classes in each, in the form [[1,2,3,4],
                                                                        [1,3,15]...]
        dataset_config: dataset_config
    """

    distances_file = pathlib.Path(dataset_config.annot_dir) / "tie_dist_leaf_only.pkl"

    # If the file is already available, return the distances.
    if distances_file.is_file():
        with open(distances_file, "rb") as infile:
            distances = pickle.load(infile)
        return distances

    hierarchy_tree = get_nltk_tree_from_hierarchy(hierarchy_int)

    leaf_nodes = []
    for subtree in hierarchy_tree.subtrees():
        if subtree.height() == 2:
            # Extending the list by the of subtrees.
            leaf_nodes.extend(subtree.leaves())
    
    tie_distances = defaultdict(tuple)

    for idx1 in tqdm(range(len(leaf_nodes))):
        for idx2 in range(idx1 + 1, len(leaf_nodes)):
            tie_dist = lca_distance(hierarchy_tree, idx1, idx2)
            tie_distances[(leaf_nodes[idx1], leaf_nodes[idx2])] = tie_dist

    # It helps other functions which infer how many levels of lca_heights are present
    tie_distances = [tie_distances]
    
    return tie_distances    

def get_lca_height_of_leaves(hierarchy_int, dataset_config):
    """
    This function returns the HEIGHT OF LCA, calculated between the leaf nodes.

    -----------------------------------------------------------------------------------
    Args:
        hierarchy_int: integer mapping of classes in each, in the form [[1,2,3,4],
                                                                        [1,3,15]...]
        dataset_config: dataset_config
    """

    distances_file = pathlib.Path(dataset_config.annot_dir) / "lca_height_leaf_only.pkl"

    # If the file is already available, return the distances.
    if distances_file.is_file():
        with open(distances_file, "rb") as infile:
            distances = pickle.load(infile)
        return distances

    hierarchy_tree = get_nltk_tree_from_hierarchy(hierarchy_int)

    leaf_nodes = []
    for subtree in hierarchy_tree.subtrees():
        if subtree.height() == 2:
            # Extending the list by the of subtrees.
            leaf_nodes.extend(subtree.leaves())

    lca_heights = defaultdict(tuple)

    for idx1 in tqdm(range(len(leaf_nodes))):
        for idx2 in range(idx1 + 1, len(leaf_nodes)):
            height = lca_height(hierarchy_tree, idx1, idx2)
            lca_heights[(leaf_nodes[idx1], leaf_nodes[idx2])] = height

    # It helps other functions which infer how many levels of lca_heights are present
    lca_heights = [lca_heights]

    # Saving the calculated distances.
    with open(distances_file, "wb") as outfile:
        pickle.dump(lca_heights, outfile)

    return lca_heights


def construct_cost_matrix(
        hierarchy_int, statistic, num_cls_each_level=None, only_last_level=True, leaves_mapping=None
):
    """
    Calculates a symmetric cost matrix where the cell i,j contains the LCA hieght between the nodes
    i and j. The same applies to any i,j.
    _______________________________________________________________________________________________
    Args:
        hierarchy_int: integer mapping of classes in each, in the form [[0,2,3],
                                                                        [0,2,15]...]

        statistic: a list of defaultdict().
                   which containes some kind of statistic, 
                   it can be Height of LCA, or it can be Tree induced error

        only_last_level: TRUE, when a cost matrix of N x N is required where N is the number of nodes
                        in the leaf level. Else, cost matrix of all levels will be returned in a list.
    """
    #  Only selecting the last level.
    if only_last_level:
        
        assert leaves_mapping is not None, "Integer mapping of leaves are not given"

        num_leaves = len(hierarchy_int)
        cost_matrix = torch.zeros((num_leaves, num_leaves), dtype=torch.float32)

        statistic = statistic[-1]
        for node_pair, distance in statistic.items():
            i, j = leaves_mapping[node_pair[0]], leaves_mapping[node_pair[1]]

            # cost matrix is a symmetric matrix, fill both i,j and j,i
            cost_matrix[i, j] = distance
            cost_matrix[j, i] = distance

        # For easier handling, we will send this cost matrix as a list
        # with only one cost matrix
        return [cost_matrix]

    else:
        cost_matrices_all_levels = []
        num_levels = len(hierarchy_int[0])

        if num_cls_each_level is None:
            raise ValueError("Please provide the number of classes in each level.")

        for i in range(num_levels):
            num_nodes = num_cls_each_level[i]
            cost_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)

            current_distance = statistic[i]
            for node_pair, distance in current_distance.items():
                i = node_pair[0]
                j = node_pair[1]
                
                # cost matrix is a symmetric matrix, fill both i,j and j,i
                cost_matrix[i, j] = distance
                cost_matrix[j, i] = distance
            cost_matrices_all_levels.append(cost_matrix)

        return cost_matrices_all_levels


def get_nltk_tree_from_hierarchy(hierarchy):
    """
    In this function, we will construct an nltk.Tree using all the unique paths from 
    parent levels to leaf, which are in the variable hierarchy. For example, to create the 
    below example tree
            1
         ___|___
        2   3   4
        |   |   |
        1   1   1
                |
                6
    
    the hierarchy, the input should be like [[1, 2, 3],[1, 3, 1],[1, 4, 1, 6]] 
    -----------------------------------------------------------------------------------
    Args:
        hierarchy: all unique paths from root to leaf, in the form [[1, 2, 33],
                                                                    [1, 3, 56], ...]
    """

    # Maximum path length tells us the number of levels
    num_levels = max([len(x) for x in hierarchy])

    # Creating a dummy_root, to construct the hierarchy
    tree = Tree("root", [])

    for path in hierarchy:
        prev_node = tree
        path_len = len(path)

        # Since the input hierarchy_int contains class names at every level
        # mapped to integers, the integers will be converted to strings
        path = [str(node_label) for node_label in path]

        # In order to differentiate between a '15' from LEVEL-3, and a possible
        # '15' from level 4, we will append the level number before the name of 
        # the node.
        path = [str(i+1) + '_' + node_label for i, node_label in enumerate(path)] 

        # If we just use integer mapping of classes in any level, there is no way of differentiating
        
        for i, node_label in enumerate(path):
            # Since not all children are Tree objects and some paths end early, we may encounter
            # a leaf node, on which .label() cannot be applied.
            children_of_prev_node = [x if type(x) == str else x.label() for x in prev_node]

            # Since the last element is the leaf node, we will just append a string, and not a tree object
            if i + 1 == path_len:
                prev_node.append(node_label)

            elif node_label in children_of_prev_node:
                # Set pointer to appropriate child.
                child_position = children_of_prev_node.index(node_label)
                prev_node = prev_node[child_position]
                continue

            else:
                new_node = Tree(node_label, [])
                prev_node.append(new_node)
                prev_node = new_node

    return tree


"""
code taken from https://github.com/cvjena/semantic-embeddings modified to work on pytorch tensors.

NOTE: in case of thousands of classes, the above repo provides an approximation algorithm, that is computationally efficient.
"""


def get_unitsphere_embeddings(class_sim):
    """
    This function finds an embedding of `n` classes on a unit sphere in `n`-dimensional space,
    so that their dot products correspond to pre-defined similarities.
    -------------------------------------------------------------------------------------------------
    Args:
        class_sim: - N x N matrix specifying the desired similarity between each pair of classes.
    Returns:
        N x N matrix with rows being the locations of the corresponding classes in the embedding space.
    """

    # Check arguments
    if (len(class_sim.shape) != 2) or (class_sim.shape[0] != class_sim.shape[1]):
        raise ValueError(
            "Given class_sim has invalid shape. Expected: (n, n). Got: {}".format(
                class_sim.shape
            )
        )
    if class_sim.shape[0] == 0:
        raise ValueError("Empty class_sim given.")

    # Place first class
    num_classes = class_sim.shape[0]
    embeddings = np.zeros((num_classes, num_classes))
    class_sim = class_sim.numpy()

    # Embedding of the first class will be [1, 0, 0, ... 0]
    embeddings[0, 0] = 1.0

    print("Constructing unit-hypersphere embeddings ...")
    # Iteratively place all remaining classes
    for c in tqdm(range(1, num_classes)):
        embeddings[c, :c] = np.linalg.solve(embeddings[:c, :c], class_sim[c, :c])
        embeddings[c, c] = np.sqrt(1. - np.sum(embeddings[c, :c] ** 2))

    return torch.tensor(embeddings, dtype=torch.float32)


def get_unitsphere_embeddings_approx(class_sim, num_dim=None):
    
    if (len(class_sim.shape) != 2) or (class_sim.shape[0] != class_sim.shape[1]):
        raise ValueError('Given class_sim has invalid shape. Expected: (n, n). Got: {}'.format(class_sim.shape))
    if (class_sim.shape[0] == 0):
        raise ValueError('Empty class_sim given.')
    
    class_sim = class_sim.numpy()
    # Compute optimal embeddings based on eigendecomposition of similarity matrix
    L, Q = np.linalg.eigh(class_sim)
    if np.any(L < 0):
        raise RuntimeError('Given class_sim is not positive semi-definite.')
    embeddings = Q * np.sqrt(L)[None,:]

    # Approximation using the eigenvectors corresponding to the largest eigenvalues
    if (num_dim is not None) and (num_dim < embeddings.shape[1]):
        embeddings = embeddings[:,-num_dim:]  # pylint: disable=invalid-unary-operand-type
    
    return embeddings
