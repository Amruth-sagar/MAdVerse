# Newspaper Advertisement Extractor
This repository contains scripts to train and run an object detection model for extracting advertisements from newspaper images. The annotations are specifically tailored for this task, with the goal of detecting and segmenting advertisements for downstream processing and analysis.

1. `visualization.py`: Perform object detection and visualization on images using a pre-trained model.
2. `trainer.py`: Train the Faster R-CNN model on the provided dataset.
3. `COCOgen.py`: Convert annotations from Labelbox into the COCO JSON format required by the model.
4. `intersection.py`: Copy JSON files and their corresponding cropped images to a separate folder for further processing.

## Prerequisites

- Python 3.x
- Required Python packages (e.g., detectron2, wandb, Pillow)

## Setup

1. Clone the repository
2. Install the required Python packages

## Usage

### 1. Generate COCO JSON file

```bash
python COCOgen.py
```

This script will generate a `COCO_File.json` file in the `../COCO_File.json` path, containing the annotations in the COCO JSON format.

### 2. Train the model

```bash
python trainer.py
```

This script will train the Faster R-CNN model on the provided dataset. It will log the training metrics to Weights & Biases (WandB) and save the trained model weights in the `../model_weights_7/` directory.

### 3. Run object detection and visualization

```bash
python visualization.py
```

This script will run object detection and visualization on the images using the trained model. It will save the detected bounding boxes as JSON files in the `../Outputs_Final/` directory.

### 4. Copy JSON files and cropped images

```bash
python intersection.py
```

This script will copy the JSON files and their corresponding cropped images from the `../Outputs` and `../Cropped` directories, respectively, to the `../final_jsons` directory.

## Notes

- Make sure to update the paths in the scripts according to your directory structure.
- The `visualization.py` script assumes the existence of a `nn_config.yaml` file and a pre-trained model weights file (`model_final.pth`) in the specified directory.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.