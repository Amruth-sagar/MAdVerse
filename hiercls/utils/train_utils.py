from tqdm import tqdm
import torch
import ipdb
from torch.nn.functional import softmax, log_softmax, kl_div
from hiercls.utils.registry import *

"""
    NOTE: The ground truth labels are 0-indexed!
"""


def train_one_epoch(
    train_dataloader, criterion, model, curr_epoch, device, optimizers, schedulers, loss_class=None
):
    gt_labels = []
    pred_labels = []
    prob_scores = []
    running_batch_loss = 0

    model.train()

    print("TRAIN EPOCH:{0}".format(curr_epoch))
    for batch in tqdm(train_dataloader):
        images, labels = batch

        labels = labels.to(device)
        images = images.to(device)

        branch_outputs = model(images)
        
        loss = criterion(branch_outputs, labels)

        for optimizer in optimizers:
            optimizer.zero_grad()

        loss.backward()

        for optimizer in optimizers:
            optimizer.step()

        batch_preds, batch_probs = calculate_batch_predictions(branch_outputs, criterion, loss_class, device)
        
        gt_labels.extend(labels.tolist())

        pred_labels.extend(batch_preds)
        
        prob_scores.append(batch_probs)

        # scaling the batch loss with batch length
        running_batch_loss += loss.item() * len(images)

        # Stepping learning rate.
        if schedulers is not None:
            for scheduler in schedulers:
                scheduler.step()

    # The number of tensors in the probs_epoch depends on the number of classifiers (num_levels)
    # Hence, size of any element in pred_label is num_levels (or 1) depending on architecture
    probs_epoch = []
    
    if loss_class not in ['barz_denzler_l']:
        for level in range(len(pred_labels[0])):
            probs_epoch.append(torch.vstack([batch_probs[level] for batch_probs in prob_scores]))

    # Normalizing the running loss with the dataset length
    epoch_loss = running_batch_loss / len(train_dataloader.dataset)

    return (epoch_loss, pred_labels, gt_labels, probs_epoch)


def valid_one_epoch(
    valid_dataloader, criterion, model, curr_epoch, device, is_test=False, loss_class=None
):
    gt_labels = []
    pred_labels = []
    prob_scores = []
    running_batch_loss = 0

    model.eval()

    if not is_test:
        print("VAL EPOCH:{0}".format(curr_epoch))
    with torch.no_grad():
        for batch in tqdm(valid_dataloader):
            images, labels = batch
            
            labels = labels.to(device)
            images = images.to(device)

            branch_outputs = model(images)
            loss = criterion(branch_outputs, labels)
        
            batch_preds, batch_probs = calculate_batch_predictions(branch_outputs, criterion, loss_class, device)
            
            gt_labels.extend(labels.tolist())

            pred_labels.extend(batch_preds)
            
            prob_scores.append(batch_probs)

            # scaling the batch loss with batch length
            running_batch_loss += loss.item() * len(images)

    # The number of tensors in the probs_epoch depends on the number of classifiers (num_levels)
    # Hence, size of any element in pred_label is num_levels (or 1) depending on architecture
    probs_epoch = []

    if loss_class not in ['barz_denzler_l']:
        for level in range(len(pred_labels[0])):
            probs_epoch.append(torch.vstack([prob_sample[level] for prob_sample in prob_scores]))

    # Normalizing the running loss with the dataset length
    epoch_loss = running_batch_loss / len(valid_dataloader.dataset)
    return (epoch_loss, pred_labels, gt_labels, probs_epoch)



def test_one_epoch(
    test_dataloader,
    criterion,
    model,
    device,
    hierarchy_int,
    posthoc_methods_list = None,
    bet_on_leaf_level = False,
    loss_class = None
):
    """
    Test one epoch, with or without posthoc correction of probabilities
    ______________________________________________________________________________________
    Args:
        test_dataloader: test dataloader
        criterion: loss function
        model: model loaded with the checkpoint from train
        device: device the model is on
        hierarchy_int: contains all paths from parent level to leaf. It is in the form
                        [[0, 1, 2],
                         [1, 1, 5],
                         ...]
        posthoc_methods_list:  a list of tuples, where the first element is every tuple
                            is the name of the posthoc method, and the second element
                            is the object itself.  [('method_1', method_1_object),
                                                    ('method_2', method_2_object)
                                                    ...]
        bet_on_leaf_level: TRUE, IF we want to blindly trust the leaf level probabilities,
                            and change(or create) the predictions of parents based on the 
                            path from predicted leaf node to top levels.
    """
    gt_labels = []
    pred_labels = []
    prob_scores = []
    running_batch_loss = 0

    model.eval()

    print("TEST EPOCH")

    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            images, labels = batch

            labels = labels.to(device)    
            images = images.to(device)

            branch_outputs = model(images)
            loss = criterion(branch_outputs, labels)

            batch_preds, batch_probs = calculate_batch_predictions(branch_outputs, criterion, loss_class, device)

            # >>>>>>>>>>>>>>>>>>>>>= DELETE IF USELESS =<<<<<<<<<<<<<<<<<<
            # # Convert the gt labels to some mapped integers
            # if criterion.leaf_only:
            #     modified_labels = [str(len(label)) + '_' + str(label[-1].item()) for label in labels]
            #     gt_labels_mapped = [[criterion.node_to_int[label]] for label in modified_labels]
            #     gt_labels.extend(gt_labels_mapped)
            # else:
            
            gt_labels.extend(labels.tolist())
                
            pred_labels.extend(batch_preds)

            if posthoc_methods_list is not None:
                modified_probs = posthoc_handler(posthoc_methods_list, branch_outputs)

                # If all level's probabilities are changed
                if len(modified_probs) == len(branch_outputs):
                    batch_probs = modified_probs

                # If only leaf level probabilities are changed
                elif len(modified_probs) == 1:
                    batch_probs[-1] = modified_probs[-1]

            prob_scores.append(batch_probs)
            running_batch_loss += loss.item() * len(images)

    # We utilize leaf-level probabilities and trace path from predicted leaf node to root node, 
    # and the nodes in the path become the predictions of parent levels.
    if bet_on_leaf_level:
        extended_predictions = []

        path_from_leaf = dict(
            [(path[-1], (path[-2::-1]).tolist()) for path in hierarchy_int]
        )

        for predicted_leaf_class in pred_labels:

            # >>>>>>>>>>>>>>>>>>>>>= DELETE IF USELESS =<<<<<<<<<<<<<<<<<<
            # if criterion.leaf_only:
            #     prediction = int(criterion.int_to_node[predicted_leaf_class[-1]].split('_')[-1])

            #     path_leaf_to_root = [prediction] + path_from_leaf[prediction]
            # else:
            #     path_leaf_to_root = [predicted_leaf_class[-1]] + path_from_leaf[predicted_leaf_class[-1]]

            # 'pred_labels' is always a list of list, irrespective of number of levels. [[165],[65],..] or [[1,20,154],[3,50,401],...]
            path_leaf_to_root = [predicted_leaf_class[-1]] + path_from_leaf[predicted_leaf_class[-1]]


            # Since gt_labels are ordered from coarse to fine class, reverse the path
            path_root_to_leaf = path_leaf_to_root[::-1]
            extended_predictions.append(path_root_to_leaf)

        # >>>>>>>>>>>>>>>>>>>>>= DELETE IF USELESS =<<<<<<<<<<<<<<<<<<
        # # Also, we also have to convert back the gt_labels, since they are mapped
        # # to different numbers by mapping dict in criterion.
        # if criterion.leaf_only:
        #     new_gt = []
        #     leaf_to_labels_mapping = dict([(hier[-1], list(hier)) for  hier in hierarchy_int])

        #     for gt_label in gt_labels:
        #         remapped_gt = int(criterion.int_to_node[gt_label[-1]].split('_')[-1])
        #         remapped_gt = leaf_to_labels_mapping[remapped_gt]

        #         new_gt.append(remapped_gt)
            
        #     gt_labels = new_gt

        pred_labels = extended_predictions

    # The number of tensors in the probs_epoch depends on the number of classifiers (num_levels)
    probs_epoch = []
    if loss_class not in ['barz_denzler_l']:
        for level in range(len(prob_scores[0])):
            probs_epoch.append(torch.vstack([prob_sample[level] for prob_sample in prob_scores]))

    # Normalizing the running loss with the dataset length
    epoch_loss = running_batch_loss / len(test_dataloader.dataset)
    return (epoch_loss, pred_labels, gt_labels, probs_epoch)




"""
    Initially we experimented using applying post-hoc methods, to alter softmax probabilities which
    reduces mistake severity, but we didnt proceed in that direction as it was out of scope. 
    So, these two functions are kept for sake of not distrubing code that invoke them.
"""

def calculate_batch_predictions(branch_outputs, criterion, loss_class, device):
    """
    This function returns the batch prediction based on the type of loss used.

    The way predictions calculated vary depending on the loss type. Label embedding 
    losses might have have to calcualte predictions based on the similarity of the 
    predicted output with the ground truth label embeddings.
    ____________________________________________________________________________________
    Args:
        branch_outputs: it is a list of logits, with each logit having the shape of 
                        [batch, num_cls_in_that_level]. [[batch, num_L1],
                                                         [batch, num_L2],
                                                         ...] 
        criterion:      loss object, which is useful to get the precomputed things in some
                        losses.
        loss_class:     The name of the loss. Example "sum_ce", "dot", "barz_denzler_l"
                        etc
    Return:
        batch_preds:    tensor of shape [batch, num_levels]
        batch_probs:    softmax of logits, list of length num_levels with items in it 
                        of the shape [batch, <num_cls>]
    """
    label_embedding_based_losses = ["barz_denzler_l"]

    batch_preds = []
    batch_probs = []
    
    if loss_class not in label_embedding_based_losses:
        # batch_preds -> [batch, num_levels] 
        batch_preds = torch.vstack([logits.argmax(dim=1) for logits in branch_outputs]).T.tolist()
        batch_probs = [softmax(logits, dim=1) for logits in branch_outputs]
    
    elif loss_class in label_embedding_based_losses:
        if loss_class == 'barz_denzler_l': 
            # Calculating similarity between each sample and all the class embeddings.
            # NOTE: class embeddings are ROWS in embedding matrices.
            batch_preds = [
                torch.argmax(
                    torch.matmul(
                        branch_outputs[level],
                        criterion.level_wise_bd_embeddings[level].T.to(device)
                    ),
                    dim=1
                )
                for level in range(len(branch_outputs))
            ]
            batch_preds = torch.vstack(batch_preds).T.tolist()
    
    return batch_preds, batch_probs

"""
    Initially we experimented using applying post-hoc methods, to alter softmax probabilities which
    reduces mistake severity, but we didnt proceed in that direction. 
    So, these two functions are kept for sake of not distrubing code that invoke them.
"""
def posthoc_handler(posthoc_methods, branch_outputs):
    pass