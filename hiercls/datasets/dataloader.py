from torch.utils.data import DataLoader
import torch
import ipdb

def collate_function(batch):
    # inputs.shape = [batch, 3, H, W]
    inputs = torch.stack([x[0] for x in batch])

    labels = [x[1:] for x in batch]
    labels = torch.tensor(labels)

    return (inputs, labels)


def make_data_loader(dataset,
                     batch_size,
                     num_workers,
                     sampler=None):

    return DataLoader(dataset,
                      batch_size=batch_size,
                      collate_fn=collate_function,
                      num_workers=num_workers,
                      persistent_workers=True,
                      sampler=sampler)
