import cv2
import os
import json
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from PIL import Image

setup_logger()

cfg = get_cfg()
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
cfg.merge_from_file("./nn_config.yaml")
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.MODEL.WEIGHTS = "../model_weights_1/model_final.pth"
cfg.MODEL.DEVICE = "cuda"

predictor = DefaultPredictor(cfg)

lang_list = ['Malayalam']
prefix = '/scratch/venkata_kesav/news_papers/'

def process_images(image_directory, output_directory):
    print("Running for image: ", image_directory)
    for image in os.listdir(image_directory):
        print("Running for image: ", image)
        try:
            im = cv2.imread(os.path.join(image_directory, image))
            if im is None:
                print(f"Error loading image: {image}")
                with open("./error_log3", "a") as output_file:
                    output_file.write(f"Failed For image -> Corrupted, {os.path.basename(image)}")
                continue
            outputs = predictor(im)
            filename = os.path.basename(image)
            num_instances = len(outputs["instances"])
            for i in range(num_instances):
                class_id = outputs["instances"].pred_classes[i]
                scores = outputs["instances"].scores[i]
                bbox = outputs["instances"].pred_boxes.tensor[i].cpu().numpy().tolist()
                bbox_dict = {
                    "x_min" : bbox[0],
                    "y_min" : bbox[1],
                    "x_max" : bbox[2],
                    "y_max" : bbox[3]
                }
                instance_data = {
                    "file_name": filename,
                    "instance_id": f"{filename}_instance{i+1}",
                    "score": scores.item(),
                    "bounding_box": bbox_dict,
                }
                output_file_path = os.path.join(output_directory, f"{os.path.splitext(filename)[0]}_instance{i+1}.json")
                with open(output_file_path, 'w') as output_file:
                    json.dump(instance_data, output_file)
                print(output_file_path)
        except Exception as e:
            with open("./error_log3", "a") as output_file:
                output_file.write(f"Failed For image, {os.path.basename(image)}")

        print("Image Processed!!!")

for folder in lang_list:
    folder_path = os.path.join(prefix, folder)
    os.mkdir(os.path.join("/scratch/venkata_kesav/Outputs_Final/", folder))
    process_images(folder_path, f"/scratch/venkata_kesav/Outputs_Final/{folder}/")
