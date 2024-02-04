# Facial_Expression_Recognition
Face Emotion Recognition from Facial Expression with Real Time || Computer Vision
Project 1: Convolutional Neural Network (CNN) for Facial Expression Recognition
## Overview
This project focuses on developing a Convolutional Neural Network (CNN) for facial expression recognition using the FER dataset. The model is designed to classify facial expressions into seven categories: Angry, Disgust, Fear, Happy, Neutral, Sad, and Surprise.

## Dataset
The training and validation datasets consist of grayscale images with a target size of (48, 48). The dataset contains 58,454 training images and 7,066 validation images, distributed across seven expression classes.

# Model Architecture
The CNN architecture consists of four convolutional layers, each followed by Batch Normalization, ReLU activation, MaxPooling, and Dropout layers. The model further includes three fully connected layers with Batch Normalization, ReLU activation, and Dropout. The final output layer utilizes softmax activation for multi-class classification.

# Training
The model is trained using the Adam optimizer with a learning rate of 0.001 and categorical crossentropy loss. To prevent overfitting, a ReduceLROnPlateau callback is employed. The training process spans 30 epochs, with both training and validation performance monitored.

# Results
After training, the model achieves impressive accuracy on the validation set, showcasing its capability to recognize facial expressions accurately. The training progress is visualized with loss and accuracy plots.

# Project 2: Feature Selection using Genetic Algorithm with Xception Features
Overview
This project focuses on utilizing a Genetic Algorithm to perform feature selection for facial expression recognition. It employs the Xception model to extract features from facial images. The goal is to identify the most relevant features for improved classification performance.

Feature Extraction
Xception, pretrained on ImageNet, is used to extract features from the FER dataset. These features are then employed for subsequent feature selection.

Genetic Algorithm
A Genetic Algorithm is implemented to iteratively select the most informative features. The algorithm involves initializing a population of potential feature sets, evaluating their fitness using a random forest classifier, and iteratively evolving the population through crossover and mutation operations.

Training and Evaluation
The selected features are used to train a Random Forest classifier, and the model is evaluated on the test set. The accuracy of the model demonstrates the effectiveness of the genetic algorithm in identifying relevant features.

Instructions for Use
To use the provided code, ensure that the required dependencies are installed. Additionally, modify the file paths for the dataset in the code to match your local directory structure. The code includes comments for clarity and customization.

Note: Please refer to the documentation and comments within the code files for more detailed information and instructions.
