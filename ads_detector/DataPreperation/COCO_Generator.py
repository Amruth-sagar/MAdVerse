import json
import os
from PIL import Image

def create_coco_json(images_folder, annotations_folder, output_json_path):
    coco_data = {
        "info": {},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": []
    }

    categories = [
        {"id": 0, "name": "Advertisement", "supercategory": "Content"},
    ]
    coco_data["categories"] = categories

    image_id = 1
    for filename in os.listdir(images_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(images_folder, filename)
            image = Image.open(image_path)
            image_info = {
                "id": image_id,
                "file_name": filename,
                "width": image.width,
                "height": image.height
            }
            coco_data["images"].append(image_info)
            image_id += 1

    annotation_id = 1
    for filename in os.listdir(annotations_folder):
        if filename.lower().endswith('.json'):
            try:
                with open(os.path.join(annotations_folder, filename)) as json_file:
                    print(filename)
                    data = json.load(json_file)
                    image_filename = data["file_name"]
                    image_id = [img["id"] for img in coco_data["images"] if img["file_name"] == image_filename][0]
                    bounding_box = data["bounding_box"]

                    coco_box = [
                        bounding_box["x_min"],
                        bounding_box["y_min"],
                        bounding_box["x_max"] - bounding_box["x_min"],
                        bounding_box["y_max"] - bounding_box["y_min"]
                    ]

                    area = coco_box[2]*coco_box[3]

                    coco_annotation = {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": 0,
                        "bbox": coco_box,
                        "area": area,
                        "iscrowd": 0
                    }
                    coco_data["annotations"].append(coco_annotation)
                    annotation_id += 1
            except Exception as e:
                print("Unable to Load JSON")

    with open(output_json_path, "w") as output_json:
        json.dump(coco_data, output_json)
        print("Done")

if __name__ == "__main__":
    images_folder = "../test"
    annotations_folder = "../final_jsons"
    output_json_path = "../COCO_File.json"
    create_coco_json(images_folder, annotations_folder, output_json_path)
