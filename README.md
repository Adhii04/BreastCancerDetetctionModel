# Breast Cancer Detection using Convolutional Neural Networks (CNN)

## Project Overview

This project aims to develop a deep learning model to detect breast cancer from histopathological images. Leveraging Convolutional Neural Networks (CNNs), the model is trained to classify images as either 'benign' (no cancer) or 'malignant' (cancer present).

This initial version of the project focuses on building and evaluating a CNN model from scratch. Future iterations will explore more advanced techniques like transfer learning.

## Problem Statement

Breast cancer is one of the most common cancers among women worldwide. Early and accurate detection is crucial for effective treatment and improved patient outcomes. Histopathological images, which show tissue samples under a microscope, contain vital information for diagnosis. Automating the analysis of these images using deep learning can assist medical professionals in making faster and more consistent diagnoses.

## Dataset

The dataset consists of histopathological images categorized into two main classes:
* `benign`: Images showing no presence of cancerous cells.
* `malignant`: Images showing the presence of cancerous cells.

The dataset has a nested folder structure (e.g., `breast/benign/SOB/adenosis/SOB_B_A_14-22549AB/40X/image.png`), where `benign` and `malignant` are the top-level class folders, and images are nested several levels deep.

**Data Split:**
* Training Data: 50%
* Validation Data: 20%
* Test Data: 30%

The split is performed with stratification to ensure a balanced representation of both classes in each subset.

## Methodology (Simple CNN Model)

This project implements a custom Convolutional Neural Network (CNN) for image classification.

**Key Steps:**

1.  **Data Loading and Preprocessing:**
    * Images are loaded from their nested directories.
    * Each image is associated with its corresponding `benign` or `malignant` label based on its top-level folder.
    * Images are resized to a consistent dimension (e.g., 224x224 pixels) and pixel values are normalized to a [0, 1] range.
    * **Data Augmentation:** Techniques like rotation, shifting, shearing, zooming, and flipping are applied to the training data to increase its diversity and improve model generalization, which is crucial for medical imaging datasets.
2.  **Model Architecture:**
    * A sequential CNN model consisting of multiple `Conv2D` layers, `BatchNormalization`, `MaxPooling2D` layers, and `Dropout` layers for feature extraction.
    * A `Flatten` layer to convert the 2D feature maps into a 1D vector.
    * `Dense` (fully connected) layers with ReLU activation.
    * A final `Dense` layer with `sigmoid` activation for binary classification, outputting the probability of an image being 'malignant'.
3.  **Model Compilation:**
    * **Optimizer:** Adam optimizer.
    * **Loss Function:** `binary_crossentropy`, suitable for binary classification.
    * **Metrics:** Accuracy, Precision, Recall (Sensitivity), and AUC (Area Under the Receiver Operating Characteristic Curve).
4.  **Model Training:**
    * The model is trained using the prepared training data, with validation performed on the validation set.
    * **Early Stopping:** A callback is used to stop training if validation loss does not improve for a set number of epochs, preventing overfitting and saving computational resources.
    * **Model Checkpoint:** The best performing model (based on validation loss) during training is automatically saved.
5.  **Model Evaluation:**
    * The trained model is evaluated on the unseen test dataset.
    * Performance metrics (Accuracy, Precision, Recall, F1-Score, AUC) are calculated and reported.
    * A **Confusion Matrix** is generated to visualize true positives, true negatives, false positives, and false negatives.
    * An **ROC Curve** is plotted to assess the model's diagnostic ability across various thresholds.
6.  **Prediction Visualization:**
    * Sample predictions are visualized to show true vs. predicted labels along with the model's confidence scores.

## File Structure
