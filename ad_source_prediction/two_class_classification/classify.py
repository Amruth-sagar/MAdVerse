from argparse import ArgumentParser
from models import *
import torch
from tqdm import tqdm
import random
import os
from utils import load_config
from transformers import ViTFeatureExtractor, ConvNextFeatureExtractor
from torch.utils.data import Dataset, DataLoader
from glob import glob
import cv2
import pickle
import shutil
random.seed(42)


class AdSource(Dataset):
    def __init__(self,
                 images_dir,
                 img_preprocessor):

        assert os.path.exists(
            images_dir), "The path {} does not exist".format(images_dir)

        self.data_dirs = images_dir

        self.dataset = self.train_dataset()

        self.image_transforms = img_preprocessor

    def train_dataset(self):
        data = glob(os.path.join(self.data_dirs, '**/*.*'), recursive=True)
        random.shuffle(data)
        return data


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        filename = self.dataset[idx]

        # The shape of the image is (height, width, channels)
        # any image is read as BGR image --> converted to RGB
        image = cv2.imread(
            filename,
            flags=cv2.IMREAD_COLOR)

        # If images is empty for some I/O reasons
        if image is None: 
            return (torch.zeros(3,224,224), self.dataset[idx])

        image = cv2.cvtColor(image,
            code=cv2.COLOR_BGR2RGB)
        transformed_image = self.image_transforms(image, return_tensors="pt")[
            "pixel_values"].squeeze(0)  # 3, H, W
        return (transformed_image, self.dataset[idx])


def collate_function(batch):

    inputs = torch.stack([x[0] for x in batch])
    filenames = [x[1] for x in batch]

    # (batch_size, channels, h, w),  (batch_size)
    return (inputs, filenames)


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


config = load_config('/home2/rishabh.s/ADS_NON_ADS_Classifier/classification/config.yaml')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

argument_parser = ArgumentParser()

argument_parser.add_argument(
    '--dataset', type=str, help='Path of the un-annotated images folder', required=True
)
argument_parser.add_argument(
    '--ckpt_path', type=str, help="Path of the checkpoint", required=True
)
argument_parser.add_argument(
    '--pred_out', type=str, help="pkl file to store predictions", required=True
)
argument_parser.add_argument(
    '--out_folder', type=str, help="pkl file to store predictions", required=True
)

arguments = argument_parser.parse_args()


# The below variable tells what should be used as the feature extractor,
# like vit, or convnext etc. It is specified in config file.
feat_ext_name = config['model']['feat_extractor']
feat_ext_config = config['arch'][feat_ext_name]


# Image preprocessor is sent to AdsNonAds dataset, which will preprocess the input the same way it
# was done for the images during pretraining of that model.

# NOTE: <something>FeatureExtractor == used for preprocessing purposes.
# IT WILL NOT RETURN feature vector of an image
img_preprocessor = None

if feat_ext_name == 'vit':
    img_preprocessor = ViTFeatureExtractor.from_pretrained(
        feat_ext_config['args']['pretrained'])
else:
    img_preprocessor = ConvNextFeatureExtractor.from_pretrained(
        feat_ext_config['args']['pretrained'])

dataset = AdsNonAds(images_dir=arguments.dataset,
                    img_preprocessor=img_preprocessor)

dataloader = make_data_loader(dataset=dataset,
                              batch_size=config['model']['batch_size'],
                              num_workers=4,
                              sampler=None)

# We will use model_name_to_class, which is a dictionary that maps name
# of the model, from str to class, to initialize the desired model.

model = model_name_to_class[feat_ext_config['class']](
    pretrained=feat_ext_config['args']['pretrained'],
    feature_dim=feat_ext_config['args']['feature_dim'],
    num_classes=feat_ext_config['args']['num_classes'],
    dropout_prob=feat_ext_config['args']['dropout_prob'],
    is_trainable=feat_ext_config['args']['is_trainable']
)

ckpt = torch.load(arguments.ckpt_path)
model.load_state_dict(ckpt["model_state_dict"])
print("\nModel is loaded with the checkpoint")
model.to(device)

filenames = []
predictions = []
out_folder = arguments.out_folder

with torch.no_grad():
    model.eval()
    for batch in tqdm(dataloader):
        inputs, img_filenames = batch
        inputs = inputs.to(device)

        output = model(inputs)
        temp=output.argmax(dim=1).tolist()
        for i in range(len(output)):
            if temp[i]==1:
                shutil.copy(img_filenames[i],out_folder)
            
            
        predictions.extend(temp)
        filenames.extend(img_filenames)

with open(arguments.pred_out, 'wb') as outfile:
    pickle.dump(list(zip(filenames, predictions)), outfile)

print(len(filenames),len(predictions))
print(filenames[0],predictions[0])

# c=0
# for i, ad in tqdm(enumerate(ads)):
#     temp = ad[ad.rfind('/')+1:ad.rfind('_')]
#     t= glob(f'*/*/{temp}')
#     if len(t)!=1:
#         c+=1
#         print(ad,t)
#         continue
#     t=t[0]
#     file_name=ad[ad.rfind('/')+1:]
#     final_path = f'./{t}/{file_name}'