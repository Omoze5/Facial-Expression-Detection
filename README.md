# Project Overview: Facial Expression Detection Model
This project is an image classificaion problem which focuses on building a model to detect facial expressions using CNN in deep learning techniques.The dataset for this project was gotten from kaggle which contained 7 labeled facial expression images such as anger, disgust,happy, sad,neutral, fear and surprise. The key steps involved in the project are as follows:

### Libraries and Packages:

The project uses PyTorch for building and training the neural network model. The necessary libraries imported include torch, torch.nn, torch.optim, and torchvision.transforms.

### Google Drive Integration:

The project is executed in Google Colab, with data being accessed from Google Drive. The drive is mounted to Colab to facilitate data transfer.

### Dataset Preparation:

The dataset for facial expression detection is stored in a zipped file in Google Drive. This file is unzipped, and the data paths for training and testing sets are defined.
The images are preprocessed using transformations such as converting to tensors, resizing, normalizing, and applying random horizontal flips.

### Data Loading:

The project uses the ImageFolder class from torchvision.datasets to load the training and testing datasets. These datasets are then loaded into data loaders with specified batch sizes and other parameters to facilitate easy data handling during model training and evaluation.

### Model Architecture:

A model was built using CNN from scratch. The project adapts this model for facial expression detection by modifying the final layers to suit the classification of different facial expressions.

### Training and Evaluation:

The training process involves using an optimizer from torch.optim and a loss function suitable for classification tasks. The model is trained over multiple epochs with the data loaders feeding batches of images to the model.
The evaluation metrics are computed to assess the performance of the model on the test dataset.

### Result:

The project aims to output the accuracy and other performance metrics of the model in detecting various facial expressions. During the training, the model experienced overfitting which was avoided using Batch Normaliaion and the weight decay parameer.
