from torch.utils.data import DataLoader, SubsetRandomSampler
import torch
import random


def collate_function(batch, augment=False):
    if augment:
        inputs = []
        labels = []
        for img_list, label_list  in batch:
            inputs.extend(img_list)
            labels.extend(label_list)
        zipped = list(zip(inputs, labels))
        random.shuffle(zipped)
        inputs, labels = zip(*zipped)

        inputs = torch.stack(inputs)
        labels = torch.tensor(labels)
    else:
        inputs = torch.stack([x[0] for x in batch])
        labels = torch.tensor([x[1] for x in batch])

    # (batch_size, channels, h, w),  (batch_size)
    return (inputs, labels)


def make_data_loader(dataset,
                     batch_size,
                     num_workers,
                     sampler=None,
                     data_augment=False):

    return DataLoader(dataset,
                      batch_size=batch_size,
                      collate_fn=lambda x: collate_function(x, data_augment),
                      num_workers=num_workers,
                      persistent_workers=True,
                      sampler=sampler)
