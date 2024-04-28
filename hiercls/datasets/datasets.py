from torch.utils.data import Dataset
from pathlib import Path
from os.path import join
import cv2
import random
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torchvision.transforms import InterpolationMode
from hiercls.utils.registry import registry
from hiercls.datasets import h_ads
import pickle
import ipdb


BICUBIC = InterpolationMode.BICUBIC


class BaseDataset(Dataset):
    def __init__(self, dataset_config=None, split=None, annot_file_extension='pkl'):
        """
        dataset_config: OmagaConf or similar configuration object
        """
        self.dataset_config = dataset_config

        self.seed = dataset_config.seed
        self.image_dimension = tuple(dataset_config.image_dimension)
        self.hier_annot_file = "hier_annot_{}.{}".format(split, annot_file_extension)

        self.img_transform = Compose(
            [
                ToTensor(),
                Resize(self.image_dimension, interpolation=BICUBIC),
                Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

        if annot_file_extension == 'pkl':
            with open(Path(dataset_config.annot_dir) / self.hier_annot_file, 'rb') as infile:
                self.images_and_hier_labels = pickle.load(infile)
        else:
            with open(Path(dataset_config.annot_dir) / self.hier_annot_file, "r") as infile:
                self.images_and_hier_labels = infile.read().splitlines()

        random.seed(self.seed)
        random.shuffle(self.images_and_hier_labels)

    def __len__(self):
        return len(self.images_and_hier_labels)
    
    
@registry.add_dataset_to_registry('hier_ads')
class MAdVerseDataset(BaseDataset):
    def __init__(self, dataset_config, split):
        super().__init__(dataset_config,  split, annot_file_extension='pkl')

    def __getitem__(self, index):
        annotation = self.images_and_hier_labels[index]
        image_path = Path(annotation[0])

        level_wise_labels = annotation[1:]

        try:
            image = cv2.cvtColor(
                cv2.imread(image_path.as_posix(), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB
            )
        except:
            print(image_path)
            raise ValueError

        # Image shape is (3,H,W)
        return self.img_transform(image), *level_wise_labels
