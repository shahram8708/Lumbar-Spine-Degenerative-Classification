# Lumbar Spine Degenerative Classification

## Overview
This repository contains the implementation for the RSNA 2024 Lumbar Spine Degenerative Classification competition. The goal of this competition is to create models that can classify lumbar spine degenerative conditions using MRI images. The project involves preprocessing MRI images, training a convolutional neural network (CNN) using a pre-trained ResNet50 model, and generating predictions for test images.

## Table of Contents
- [Overview](#overview)
- [Data](#data)
  - [Directory Structure](#directory-structure)
  - [train.csv](#traincsv)
  - [test.csv](#testcsv)
- [Model](#model)
  - [Preprocessing](#preprocessing)
  - [Model Architecture](#model-architecture)
  - [Training](#training)
- [Prediction and Submission](#prediction-and-submission)
- [Results](#results)
- [Usage](#usage)
- [Requirements](#requirements)
- [Acknowledgements](#acknowledgements)

## Data
### Directory Structure
The expected directory structure for the data is as follows:
```
.
├── images
│   ├── train
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── test
│       ├── test_image1.jpg
│       ├── test_image2.jpg
│       └── ...
├── train.csv
├── test.csv
└── submission.csv
```

### train.csv
The `train.csv` file contains the training data with columns:
- `row_id`: Unique identifier for the image.
- `image_path`: Path to the image file.
- `normal_mild`: Binary label indicating normal/mild condition.
- `moderate`: Binary label indicating moderate condition.
- `severe`: Binary label indicating severe condition.

### test.csv
The `test.csv` file contains the test data with columns:
- `row_id`: Unique identifier for the image.
- `image_path`: Path to the image file.

## Model
### Preprocessing
The preprocessing involves reading the images, resizing them to 224x224 pixels, and normalizing the pixel values. The preprocessing function used is:
```python
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    image = image / 255.0
    return image
```

### Model Architecture
The model is based on a pre-trained ResNet50 architecture, with custom dense and dropout layers added on top. The architecture includes:
- A ResNet50 base model with pre-trained ImageNet weights.
- Flatten layer to convert the feature maps to a 1D vector.
- Dense layer with 128 units and ReLU activation.
- Dropout layer with 50% dropout rate.
- Output Dense layer with softmax activation for 3 classes.

### Training
The model is trained in two phases:
1. **Initial Training**: The ResNet50 base layers are frozen, and only the custom top layers are trained.
2. **Fine-tuning**: The last 10 layers of the ResNet50 base model are unfrozen and re-trained with a lower learning rate.

Data augmentation is applied during training to improve generalization. The ImageDataGenerator is configured with various augmentations like rotation, width shift, height shift, shear, zoom, and horizontal flip.

## Prediction and Submission
After training, the model is used to generate predictions on the test set. The predictions are saved in the required format for submission to the competition. The submission file includes columns:
- `row_id`: Unique identifier for the image.
- `normal_mild`: Predicted probability for the normal/mild condition.
- `moderate`: Predicted probability for the moderate condition.
- `severe`: Predicted probability for the severe condition.

## Results
The model is evaluated on the validation set, and the validation accuracy is reported. The script also generates the `submission.csv` file for competition submission.

## Usage
To run the code, follow these steps:
1. Place the training and test images in the `images/train` and `images/test` directories, respectively.
2. Ensure `train.csv` and `test.csv` are properly formatted and placed in the root directory.
3. Run the script:
   ```bash
   python script.py
   ```
4. The `submission.csv` file will be generated in the root directory.

## Requirements
- Python 3.7+
- pandas
- numpy
- opencv-python
- tensorflow
- scikit-learn

Install the required packages using:
```bash
pip install pandas numpy opencv-python tensorflow scikit-learn
```

## Acknowledgements
This project is part of the RSNA 2024 Lumbar Spine Degenerative Classification competition hosted by the Radiological Society of North America (RSNA) in collaboration with the American Society of Neuroradiology (ASNR). Special thanks to the dataset contributors and annotators for their efforts in creating a comprehensive and high-quality dataset for this challenge.

For more information on the competition, visit the [Kaggle competition page](https://kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification).
