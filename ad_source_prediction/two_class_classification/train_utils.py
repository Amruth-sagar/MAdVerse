from tqdm import tqdm
import numpy as np
import torch
import random


def train_one_epoch(train_dataloader, criterion, model, curr_epoch, device, optimizers, schedulers):

    model.train()

    gt_labels = []
    pred_labels = []
    running_batch_loss = 0

    print("EPOCH:{0}".format(curr_epoch))
    for batch in tqdm(train_dataloader):
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)

        for optimizer in optimizers:
            optimizer.zero_grad()

        output = model(inputs)
        loss = criterion(output, labels)
        loss.backward()

        for optimizer in optimizers:
            optimizer.step()
        
        gt_labels.extend(labels.tolist())
        pred_labels.extend(output.argmax(dim=1).tolist())

        # scaling the batch loss with batch length
        running_batch_loss += loss.item() * inputs.shape[0]
    
        # Stepping learning rate.
        if schedulers is not None:
            for scheduler in schedulers:
                scheduler.step()

    # Normalizing the running loss with the dataset length
    epoch_loss = running_batch_loss / len(train_dataloader.dataset)

    return (epoch_loss, pred_labels, gt_labels)
        

def valid_one_epoch(valid_dataloader, criterion, model, curr_epoch, device):

    gt_labels = []
    pred_labels = []
    running_batch_loss = 0

    print("EPOCH:{0}".format(curr_epoch))
    with torch.no_grad():
        model.eval()
        for batch in tqdm(valid_dataloader):
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)

            output = model(inputs)
            loss = criterion(output, labels)

            # Scaling the batch loss with batch length
            running_batch_loss += loss.item() * inputs.shape[0]

            gt_labels.extend(labels.tolist())
            pred_labels.extend(output.argmax(dim=1).tolist())
    
    # Normalizing the running loss with the dataset length
    epoch_loss = running_batch_loss / len(valid_dataloader.dataset)
    return (epoch_loss, pred_labels, gt_labels)



def seed_everything(seed=0, harsh=True):
    """
    Seeds all important random functions
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