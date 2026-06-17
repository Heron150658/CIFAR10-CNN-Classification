# CIFAR10-CNN-Classification
A TensorFlow-based CNN implementation for CIFAR-10 image classification, including data preprocessing, model training, performance evaluation, and result visualization. This project was developed as part of an Artificial Intelligence course to demonstrate the application of deep learning in computer vision.
# Project Structure
```
CNN-CIFAR10/
│
├── CNN.py                          # Main training & evaluation script
├── README.md                       # Project documentation
│
├── saved_models/                  # Saved trained models
│   └── keras_cifar10_trained_model.h5
│
├── results/                       # Generated evaluation outputs
│   ├── classification_report.txt
│   ├── confusion_matrix.png
│   ├── accuracy_curve.png
│   └── loss_curve.png
│
└── figures/                       # Optional report figures
    ├── training_process.png
    └── sample_predictions.png
```

---
# Environment Requirements
Hardware (recommended)
| Component | Requirement                                       |
| --------- | ------------------------------------------------- |
| CPU       | Intel i5 / i7 or higher                           |
| RAM       | ≥ 8GB (recommended 16GB)                          |
| GPU       | Optional (NVIDIA CUDA supported for acceleration) |
# Software
| Package      | Version |
| ------------ | ------- |
| Python       | 3.9     |
| TensorFlow   | 2.15.0  |
| Keras        | 2.15    |
| NumPy        | ≥1.21   |
| Matplotlib   | 3.8.4   |
| Scikit-learn | 1.6.1   |
# Installation
1. Create virtual environment (recommended)
  conda create -n cnn python=3.9 -y
  conda activate cnn
2. Install dependencies
  pip install tensorflow==2.15.0
  pip install numpy matplotlib scikit-learn
# Dataset Description
The project uses the CIFAR-10 dataset, which includes:
60,000 color images (32×32 RGB)
10 categories
50,000 training images
10,000 test images
Dataset will be automatically downloaded when running the code:
https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
# How to Run
1. Train the model
   Run the main script:
   python CNN.py
   During execution, the program will:
  Load CIFAR-10 dataset
  Build CNN model
  Train for 5 epochs
  Evaluate test performance
  Save trained model
2. Output results
  After training, the following outputs will be generated:
(1) Model file
  saved_models/keras_cifar10_trained_model.h5
(2) Console outputs
  Training accuracy & loss per epoch
  Test accuracy
  Classification report
(3) Evaluation metrics
  Precision
  Recall
  F1-score
  Confusion matrix
# Model Architecture
The CNN model consists of:
2 Convolution layers (32 filters)
MaxPooling layer
Dropout layer
2 Convolution layers (64 filters)
MaxPooling layer
Flatten layer
Fully Connected layer (512 neurons)
Output layer (Softmax, 10 classes)
# Experimental Results
Final Performance
| Metric           | Value     |
| ---------------- | --------- |
| Test Accuracy    | ~70%      |
| Test Loss        | ~0.87     |
| Model Parameters | 1,250,858 |
# Training Behavior
Accuracy increases steadily from ~38% → ~70%
Loss decreases consistently during training
Slight overfitting appears after epoch 4
# Common Issues
1. TensorFlow warnings
   These are NOT errors:
     oneDNN custom operations
     deprecated API warnings
  They do not affect results.
2. Slow training
  If CPU-only:
  Training may take 30–60 seconds per epoch
3. Model not saving
   Ensure folder exists:
   mkdir saved_models
# Notes for Report Writing
You can use this project for:
  CNN algorithm reproduction experiment
  Image classification baseline model
  Deep learning coursework report
 # License
 This project is for academic use only.
 # Author
   Zhang Haoran
  Information Science and Technology
  Jinan University
