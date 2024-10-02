# Heart Disease Classification Using ECG Signals

## Project Overview

This project aims to classify heart diseases using electrocardiogram (ECG) signals with the help of various machine learning and deep learning models. The dataset is the **PTB database**, which consists of ECG recordings from multiple subjects with diverse diagnostic categories. The classification models were designed to distinguish between heart conditions, focusing on myocardial infarction and healthy controls. 

## Data Collection

- **PTB Database**: Contains **549 ECG records** from **290 patients**, categorized into various heart conditions such as myocardial infarction, cardiomyopathy, heart failure, and others.
- After preprocessing and filtering, the final dataset used for model training included subjects diagnosed with:
  - Myocardial infarction: 148 subjects
  - Healthy controls: 52 subjects
- In total, there are 368 records for myocardial infarction and 80 records for healthy controls.

## Data Preprocessing

To ensure uniformity and accuracy during feature extraction, the following preprocessing steps were applied:
- **Standardization**: ECG signals were standardized to a uniform length of **120 seconds**.
- **Filtering**: Noise and artifacts in the signals were filtered out using **Finite Impulse Response (FIR) bandpass filtering**.
- **Normalization**: Each ECG signal was normalized between **-1 and 1**.

### Feature Extraction

Instantaneous Frequency (IF) was extracted using the Hilbert transform:
\[ IF(t) = rac{1}{2\pi} rac{d\phi}{dt} \]
Where \(\phi\) represents the instantaneous phase of the ECG signal.

## Machine Learning Models

Several machine learning models were trained on the preprocessed ECG data. The models evaluated included:
- **K-Nearest Neighbors (KNN)**
- **Support Vector Machine (SVM)**
- **Decision Tree**
- **Random Forest**

The models were evaluated using **10-fold cross-validation** to ensure robustness. Key metrics such as sensitivity, precision, F1-score, accuracy, and loss were considered in model evaluation.

Additionally, **SMOTE (Synthetic Minority Over-sampling Technique)** was applied to handle the class imbalance in the dataset, improving the models' performance.

**Dimensionality Reduction and Feature Selection** were also explored using **PCA (Principal Component Analysis)** and Random Forest-based feature importance methods to further enhance model performance.

## Deep Learning Models

In addition to traditional machine learning methods, deep learning techniques were also applied.

The **ConvNetQuacke** model was designed as follows:
- **Model architecture**:
  - 1D Convolutional layers
  - Batch Normalization and MaxPooling layers
  - Dense layers with Dropout for regularization
  - Final classification with Sigmoid activation for binary classification
- **Training**:
  - The model was trained using **binary crossentropy** loss and **Adam optimizer**.
  - Used **oversampled data** (via Random Oversampling and SMOTE) to address class imbalance.
- **Evaluation**:
  - The model achieved a sensitivity of **94.64%**, precision of **92.99%**, and accuracy of **89.74%**.

Other deep learning models tested included:
- **Variational Autoencoder**
- **LSTM/GRU**
- **CNN**
- **LSTM**

## Summary of Findings

- Both machine learning and deep learning models were effective in classifying heart diseases using ECG signals.
- The **Random Forest** and **ConvNetQuacke** models showed the best overall performances.
- The dataset, after preprocessing, provided robust insights for model training, especially with the clear distinction between myocardial infarction and healthy controls.


## Instructions to Run the Project

1. **Dependencies**:
   - Libraries: `scikit-learn`, `TensorFlow`, `Keras`, `NumPy`, `Matplotlib`, `SciPy`, `h5py`, `SMOTE`, `imblearn`, and many others

2. **Steps**:
   - Clone this repository.
   - Preprocess the ECG data using the provided preprocessing scripts.
   - Train the machine learning and deep learning models using the preprocessed data.
   - Evaluate the models using the provided evaluation metrics.

