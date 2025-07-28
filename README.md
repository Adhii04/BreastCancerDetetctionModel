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

1.  **Prepare Your Dataset:**
    * Download the **BreakHis Histological Images of Breast Tumors** dataset from Kaggle: [https://www.kaggle.com/datasets/ambarish/breakhis]
    * Upload the dataset to your Google Drive. Ensure the `breast` folder (containing `benign` and `malignant` subfolders with nested images) is correctly placed.
    * Note down the exact path to this `breast` folder within your Google Drive (e.g., `My Drive/breast`).

2.  **Open the Notebook:**
    * Upload the `breast_cancer_detection.ipynb` file to your Google Colab environment.

3.  **Mount Google Drive:**
    * Run the first code cell in the notebook, which will `prompt` you to `mount` your `Google Drive`. `Follow` the `instructions` to `authorize` `access`.

4.  **Adjust `root_data_dir`:**
    * Locate the line `root_data_dir = '/content/drive/MyDrive/breast'` in the notebook.
    * **`Crucially`, `verify` and `adjust` this `path`** if your `breast` `folder` is located elsewhere in your `Google Drive`. For example, if it's in a `subfolder` `called` `Medical_Datasets`, the `path` would be `/content/drive/MyDrive/Medical_Datasets/breast`.

5.  **Run All Cells:**
    * `Execute` all `code cells` in the `notebook sequentially` (e.g., using `Runtime` -> `Run all` or by `pressing Shift` + `Enter` on each cell).
    * `Monitor` the `output`, especially the `Debugging Directory Structure` `section`, to `confirm` that all your `images` (both `benign` and `malignant`) are `correctly found`.

6.  **Review Results:**
    * The `training` `progress` (`epoch-by-epoch` `metrics`) will be displayed.
    * After `training`, the `final evaluation` `metrics` (`Loss`, `Accuracy`, `Precision`, `Recall`, `AUC`) on the `test set` will be `printed`.
    * The `Confusion Matrix` and `ROC Curve` `plots` will be `generated`.
    * `Sample prediction visualizations` will `show individual image classifications`.

## Key Dependencies

The `project relies` on the `following Python libraries`. All of them are typically `pre-installed` in `Google Colab` `environments`. If any are `missing`, you can `install` them using `!pip install <library_name>` in a `Colab cell`.

* `numpy`
* `pandas`
* `matplotlib`
* `seaborn`
* `scikit-learn`
* `tensorflow`
* `opencv-python` (`cv2`)
* `pydicom` (only required if your dataset includes `.dcm` DICOM files)

## Evaluation Metrics

The `model's` performance is `assessed` using:

* `Accuracy`: Overall `proportion` of `correct` `predictions`.
* `Precision` (for `Malignant`): The `ratio` of `true positive predictions` to the total `predicted positives`. Important to minimize `false alarms`.
* `Recall` / `Sensitivity` (for `Malignant`): The `ratio` of `true positive predictions` to the total `actual positive cases`. (**`This is a critical metric in cancer detection`.**)
* `F1-Score` (for `Malignant`): The `harmonic mean` of `Precision` and `Recall`, `providing` a `balance` between the `two`.
* `AUC` (`Area Under the Receiver Operating Characteristic` `Curve`): A `comprehensive measure` that `evaluates` the `model's` `ability` to `distinguish` between `classes` across all `possible classification thresholds`.

## Future Work

* **Transfer Learning:** Implement and `evaluate pre-trained models` (e.g., MobileNetV2, `ResNet50`, `VGG16`) using `transfer learning` `techniques`. `This is often more effective` for `medical imaging` due to `limited data`.
* **Hyperparameter Optimization:** `Utilize automated tools` (like Keras Tuner, `Optuna`) to `systematically find optimal hyperparameters` for the `model`, rather than `manual tuning`.
* **Class Imbalance Handling:** Explore `advanced` `strategies` (e.g., `weighted loss`, `oversampling`, `undersampling`) if the `dataset exhibits` `significant class imbalance`.
* **Model Interpretability:** Apply `techniques` (e.g., `Grad-CAM`) to `visualize` `which regions` of the `image` the `CNN focuses` on when `making` a `prediction`, `enhancing` `trust` and `understanding` for `medical professionals`.
* **Deployment:** `Investigate methods` for `deploying` the `trained model` as an `API` for `real-time` `inference`.
