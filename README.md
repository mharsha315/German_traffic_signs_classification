# Traffic Signs Classification

This repository contains a Python script for classifying traffic signs using a machine learning model. The script trains a model on a dataset of labeled traffic signs and predicts the category of a given image. It aims to provide a solution for automated traffic sign recognition, which can be applied in autonomous vehicles and traffic monitoring systems.

## Table of Contents

- [Features](#features)
- [Usage](#usage)
- [Dataset](#dataset)
- [Results](#results)


## Features

- Loads a dataset of labeled traffic sign images.
- Preprocesses images for model training.
- Trains a deep learning model for traffic sign classification.
- Evaluates the model's accuracy on test data.
- Predicts the class of a given traffic sign image.


## Usage

To use the traffic sign classification script, follow these steps:

1. **Prepare the Dataset**
   - Ensure that the dataset is downloaded and organized with images in folders corresponding to each class label.
   - Update the file paths in the script to match your dataset directory structure.

2. **Install Dependencies**
   - Install the required Python libraries using:
     ```bash
     pip install -r requirements.txt
     ```

3. **Run the Script**
   - Open a terminal and navigate to the project directory.
   - Execute the following command to start training the model:
     ```bash
     python traffic_signs_classification.py
     ```

4. **Testing and Evaluation**
   - The script will automatically split the data into training and testing sets, train the model, and display accuracy and loss metrics for each epoch.
   - After training, it will evaluate the model on the test dataset and display the results.

5. **Make Predictions**
   - To predict the class of a new traffic sign image, update the script with the image's file path and rerun it.
   - The predicted class will be displayed in the console output.

6. **Adjust Model Hyperparameters**
   - Modify the hyperparameters (e.g., learning rate, batch size, epochs) in the script to optimize model performance based on your dataset.

7. **Visualize Results**
   - The script will plot accuracy and loss graphs to help analyze model performance over epochs.
   - You can also enable confusion matrix visualization to get a detailed breakdown of model predictions across classes.

## Dataset

The dataset used for training and testing the traffic sign classification model consists of images categorized into different classes, each representing a specific traffic sign type. The dataset is from Kaggle.

## Results

After training the model, the script will display performance metrics and predictions to evaluate the effectiveness of the traffic sign classification.

### Model Evaluation

- **Accuracy:** The model's accuracy will be displayed at the end of training, indicating how well it classifies the traffic signs in the test dataset.
- **Loss:** The loss value will be shown, helping to understand the model's learning efficiency and error rate.

### Sample Output

- The script will print classification results, including:
  - **Training Accuracy & Loss:** Accuracy and loss values for the training phase.
  - **Validation Accuracy & Loss:** Accuracy and loss values for the validation phase.
  - **Test Accuracy:** Accuracy of the model when applied to the test dataset.

### Predictions

- The model will make predictions for the test images, and the script will display:
  - The predicted class label.
  - The actual class label (for comparison).
  - An image preview of the test sample (if configured in the script).

### Visualization

- **Accuracy & Loss Plots:** The script may generate and display plots for accuracy and loss over epochs to visualize the training progress.
- **Confusion Matrix:** If enabled, a confusion matrix can be plotted to show the model's performance across all classes.









