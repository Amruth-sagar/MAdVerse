from sklearn.manifold import TSNE
from numpy import reshape
import seaborn as sns
import pandas as pd 
import argparse
from glob import glob
import pickle
import os
import cv2
from transformers import ViTFeatureExtractor, ConvNextFeatureExtractor
import numpy as np
from tqdm import tqdm
import shutil
import json
import matplotlib.pyplot as plt

def get_tsne_plots(filenames,labels, out_dir):
    images =[]
    for filename in tqdm(filenames):
        image =cv2.imread(filename)
        if image is None:
            print("The file {} does not exist".format(filename))
            os.remove(filename)
            return (None,-1)
        image  = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_preprocessor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
        transformed_image = img_preprocessor(image, return_tensors="pt")["pixel_values"].squeeze(0)
        images.append(transformed_image)
    print(images[0].shape)
    with open('temp_three.pickle', 'wb') as f:
        pickle.dump(images, f)
    images = np.vstack(images)
    images = reshape(images, (len(labels), -1))
    tsne = TSNE(n_components=2, random_state=1)
    z = tsne.fit_transform(images)
    df = pd.DataFrame()
    df["y"] = labels
    df["comp-1"] = z[:,0]
    df["comp-2"] = z[:,1]

    splot = sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                    palette=sns.color_palette("hls", 3),
                    data=df)
    
    plt.rcParams["figure.figsize"] = [7.50, 3.50]
    plt.rcParams["figure.autolayout"] = True
    splot.figure.savefig(os.path.join(out_dir, 'tsne_gt.png'))
    
        
def get_correct_incorrect_predictions(data, out_dir):
    if os.path.exists(os.path.join(out_dir, 'correct')) == False:
        os.makedirs(os.path.join(out_dir, 'correct'))
    if os.path.exists(os.path.join(out_dir, 'incorrect')) == False:
        os.makedirs(os.path.join(out_dir, 'incorrect'))
    for i in tqdm(range(len(data['filenames']))):
        if data['gt_test'][i] == data['pred_test'][i]:
            if not os.path.exists(os.path.join(out_dir, 'correct')+'/class_'+str(data['gt_test'][i])):
                os.makedirs(os.path.join(out_dir, 'correct')+'/class_'+str(data['gt_test'][i]))
                
                
            shutil.copy(data['filenames'][i], os.path.join(out_dir, 'correct')+'/class_'+str(data['gt_test'][i])+'/')
        else:
            if not os.path.exists(os.path.join(out_dir, 'incorrect')+'/class_'+str(data['gt_test'][i])):
                os.makedirs(os.path.join(out_dir, 'incorrect')+'/class_'+str(data['gt_test'][i]))
                
            shutil.copy(data['filenames'][i], os.path.join(out_dir, 'incorrect')+'/class_'+str(data['gt_test'][i])+'/')
        
        
    
    
    
    
    

arguments_parser = argparse.ArgumentParser()

arguments_parser.add_argument('--pkl_file', help="Path of pkl_file dataset", type=str, required=True)
arguments_parser.add_argument('--output_dir', help="Path of output dataset", type=str, required=True)

arguments = arguments_parser.parse_args()
out_dir = arguments.output_dir
if os.path.exists(arguments.output_dir) == False:
    os.makedirs(arguments.output_dir)

with open(arguments.pkl_file, 'rb') as f:
    data = pickle.load(f)
filenames = data['file_names']
gt_labels = data['gt_test']
pred_labels = data['pred_test']
get_tsne_plots(filenames,gt_labels, out_dir)
# get_correct_incorrect_predictions(data, out_dir)



