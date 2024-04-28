from argparse import ArgumentParser
from dataloader import *
from datasets import *
from dataset import *
from models import *
from transformers import ViTModel
import torch
from tqdm import tqdm
import pickle
from utils import *
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from transformers import ViTFeatureExtractor, ConvNextFeatureExtractor


config = load_config('/home2/rishabh.s/AD_source_Prediction/three_class_classification/config.yaml')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

argument_parser = ArgumentParser()

argument_parser.add_argument(
    '--ckpt_path', type=str, help="Path of the checkpoint", required=True
)
argument_parser.add_argument(
    '--pred_out', type=str, help="pkl file to store predictions", required=True
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
    img_preprocessor = ViTFeatureExtractor.from_pretrained(feat_ext_config['args']['pretrained'])
else:
    img_preprocessor = ConvNextFeatureExtractor.from_pretrained(feat_ext_config['args']['pretrained'])

test_dataset = AdSource(images_dirs=[config['data']['test_dir']],
                        img_preprocessor=img_preprocessor,
                        of_num_imgs=None,
                        overfit_test=False,
                        augment_data=config['data']['augment'])
test_dataloader = make_data_loader(dataset=test_dataset,
                                    batch_size=config['model']['batch_size'],
                                    num_workers=1,
                                    sampler=None,
                                    data_augment=config['data']['augment'])

# We will use model_name_to_class, which is a dictionary that maps name
# of the model, from str to class, to initialize the desired model.

model = model_name_to_class[feat_ext_config['class']](
    pretrained = feat_ext_config['args']['pretrained'], 
    feature_dim = feat_ext_config['args']['feature_dim'] , 
    num_classes = feat_ext_config['args']['num_classes'], 
    dropout_prob = feat_ext_config['args']['dropout_prob'], 
    is_trainable = feat_ext_config['args']['is_trainable']
)

ckpt = torch.load(arguments.ckpt_path)
model.load_state_dict(ckpt["model_state_dict"])
print("\nModel is loaded with the checkpoint")
model.to(device)

gt_labels = []
pred_labels = []
file_names = []
test_loss = 0


count_trainable_parameters(model)

criterion = nn.CrossEntropyLoss()

print(len(test_dataset), config['data']['test_dir'])

with torch.no_grad():
    model.eval()
    for batch in tqdm(test_dataloader):
        try:
            inputs, labels, file_name = batch
            inputs, labels = inputs.to(device), labels.to(device)

            output = model(inputs)
            loss = criterion(output, labels)

            # Running loss scaled with batch size
            test_loss += loss.item() * inputs.shape[0]

            gt_labels.extend(labels.tolist())
            pred_labels.extend(output.argmax(dim=1).tolist())
            file_names.extend(file_name)
        except Exception as e:
            print(e)
            continue

# Normalizing the running loss with dataset length
test_loss = test_loss / len(test_dataloader.dataset)

test_accuracy = accuracy_score(y_true=gt_labels, y_pred=pred_labels)
f1_score_each_class = f1_score(y_true=gt_labels, y_pred=pred_labels, average=None)
precision_each_class = precision_score(y_true=gt_labels, y_pred=pred_labels, average=None)
recall_each_class = recall_score(y_true=gt_labels, y_pred=pred_labels, average=None)
final_result = {'feature_extractor': config['model']['feat_extractor'],
                'file_names':file_names,
                'gt_test':gt_labels,
                'pred_test':pred_labels,
                'test_loss':test_loss,
                'test_accuracy':test_accuracy,
                'test_f1_scores':f1_score_each_class,
                'test_precision':precision_each_class,
                'test_recall':recall_each_class
}

with open(arguments.pred_out, 'wb') as outfile:
    pickle.dump(final_result, outfile)

print("LOSS: {0}, ACC: {1}".format(test_loss, test_accuracy))
print("F1_SCORES: {}".format(f1_score_each_class))
