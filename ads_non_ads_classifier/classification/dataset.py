from glob import glob
from torch.utils.data import Dataset
import torchvision.transforms as T
import os
import random
import cv2
import torchvision.transforms as T

class AdsNonAds(Dataset):
    def __init__(self,
                 images_dirs,
                 img_preprocessor,
                 of_num_imgs=20,
                 overfit_test=False,
                 augment_data=False,
                 full=0):

        assert type(images_dirs) == list, "Give directory paths in a list, like ['/../dir1'] or ['/../dir1', '/../dir2', ...]"
        
        for directory in images_dirs:
            assert os.path.exists(directory), "The path {} does not exist".format(directory)

        self.data_dirs = images_dirs
        self.augment = augment_data
    
        if overfit_test:
            self.dataset = self.sample_dataset(of_num_imgs)
        elif full==0:
            self.dataset = self.train_dataset()
        elif full==1:
            self.dataset = self.full_dataset()
        self.image_transforms = img_preprocessor
        print('\n\n', self.image_transforms, '\n')

    def train_dataset(self):
        ads = []
        non_ads = []

        for directory in self.data_dirs:
            #might have to change these lines
            ads.extend(glob(directory+'/ADS/*'))
            non_ads.extend(glob(directory+'/NON_ADS/*'))
        
        data = []
        data += [[x, 1] for x in ads]
        data += [[x, 0] for x in non_ads]

        random.shuffle(data)

        return data
    def full_dataset(self):
        ads = glob(f'{self.data_dirs[0]}/*/*/*/*.*')
        non_ads= glob(f'{self.data_dirs[1]}/*/*/*/*.*')
        data = []
        data += [[x, 1] for x in ads]
        data += [[x, 0] for x in non_ads]

        random.shuffle(data)

        return data
    def sample_dataset(self, num_imgs):
        ads = []
        non_ads = []

        for directory in self.data_dirs:
            ads.extend(glob(directory+'/ads/*'))
            non_ads.extend(glob(directory+'/non-ads/*'))

        ads = random.sample(ads, num_imgs)
        non_ads = random.sample(non_ads, num_imgs)

        data = []
        data += [[x, 1] for x in ads]
        data += [[x, 0] for x in non_ads]

        random.shuffle(data)
        return data

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        filename = self.dataset[idx][0]
        if os.path.exists(filename) == False:
            print("The file {} does not exist".format(filename))
            return (None,-1)
        # The shape of the image is (height, width, channels)
        # any image is read as BGR image --> converted to RGB
        image =cv2.imread(filename)
        if image is None:
            print("The file {} does not exist".format(filename))
            return (None,-1)
        image  = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = cv2.cvtColor(cv2.imread(
        #     filename,
        #     flags=cv2.IMREAD_COLOR),
        #     code=cv2.COLOR_BGR2RGB)
        transformed_image = self.image_transforms(image, return_tensors="pt")["pixel_values"].squeeze(0) # 3, H, W
        if self.augment:
            # All the transformation expect the image to be in shape of [...., H, W]
            rotated_90 = T.functional.rotate(transformed_image, angle=90)
            rotated_270 = T.functional.rotate(transformed_image, angle=270) 
            flipped = T.RandomHorizontalFlip(p=1)(transformed_image)

            return_list = [transformed_image, rotated_90, rotated_270, flipped]

            return (return_list, [self.dataset[idx][1]]*len(return_list))
        return (transformed_image, self.dataset[idx][1])
