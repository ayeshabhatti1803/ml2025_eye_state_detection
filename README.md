# ML 2025 Final Project: Eye State Detection

This project detects whether eyes are open or closed in images using a Convolutional Neural Network (CNN) built with TensorFlow/Keras.

## Project Overview

- The model classifies images into `open_eye` or `closed_eye`.
- The dataset was sourced from [HuggingFace closed-open-eyes dataset](https://huggingface.co/datasets/MichalMlodawski/closed-open-eyes), which contains mostly AI-generated images.
- The CNN has 3 convolutional layers followed by max-pooling layers, then flattened and passed through dense layers. Dropout is used to improve accuracy.  
- Binary classification uses a sigmoid output.  
- Adam optimizer is used, with early stopping and model checkpointing during training.

## Repository Contents

- `open_eyes/` : Folder with open-eye images for training  
- `closed_eyes/` : Folder with closed-eye images for training  
- `ayesha_bhatti.ipynb` : Jupyter notebook with full code and explanations  
- `ml_25_eye_off.py` : Python script for real-time eye detection  
- `how_tensors_flow.png` : Image illustrating model workflow  

> Note: The trained model file (`ayesha_bhatti.keras`) is too large for GitHub. See the download link below.

## Usage

1. Open `ayesha_bhatti.ipynb` to train or test the model.  
2. Place your own images in the `test_images/` folder to make predictions.

## Trained Model

Download the trained model (104 MB):  
[ayesha_bhatti.keras](https://github.com/ayeshabhatti1803/ml2025_eye_state_detection/releases/download/v1.0/ayesha_bhatti.keras)  

**SHA-256 checksum:**  
`2008d36c951c795fbfd3e3d2cd894ae4d9f6d7adcb4e556c24878ee8c3b7ed7d`
