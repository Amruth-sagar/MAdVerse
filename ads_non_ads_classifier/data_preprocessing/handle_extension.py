from glob import glob
import os
import pillow_avif
from PIL import Image as pil
from tqdm import tqdm
import imagecodecs as Image
import cv2
import argparse
from pathlib import Path

if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('--dataset_dir', help="dataset path of the ad images", type=str, required=True)

    arguments = argument_parser.parse_args()

    dataset_path = Path(arguments.dataset_dir)
    dataset_path_plus_pattern = dataset_path.as_posix() + '/*'
    all_paths= glob(dataset_path_plus_pattern)

    images_file=[path for path in all_paths if os.path.isfile(path)]

    print(len(images_file))

    list_of_extensions = []
    for img in tqdm(images_file):
        extension = img[img.rfind('.')+1:]
        if extension not in list_of_extensions:
            list_of_extensions.append(extension)
        if extension != 'jpg':
            try:
                if extension =='avif':
                    im = Image.imread(img)
                    os.remove(img)
                    Image.imwrite(img[:img.rfind('.')]+'.jpg', im)
                else:
                    im = cv2.imread(img)
                    os.remove(img)
                    cv2.imwrite(img[:img.rfind('.')]+'.jpg', im)
            except Exception as e:
                print('Error in',img)
                print(e)
  
    print(list_of_extensions, len(images_file))
# ffmpeg
