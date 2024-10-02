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
- **Support Vector Machine (SVM)**
- **K-Nearest Neighbors (KNN)**
- **CatBoost**
- **Decision Tree**
- **LightGBM**
- **Random Forest**
- **XGBoost**

The models were evaluated using **10-fold cross-validation** to ensure robustness. The **Random Forest** model achieved the best accuracy of **98.71%**. Other key metrics such as sensitivity, precision, F1-score, and loss were also evaluated, with a focus on ensuring reliable classification of heart conditions.

## Deep Learning Models

In addition to traditional machine learning methods, deep learning techniques were also applied. The deep learning models tested included:
- **ConvNetQuacke**
- **Variational Autoencoder**
- **LSTM/GRU**
- **CNN**
- **LSTM**

The **ConvNetQuacke** model demonstrated the best performance, achieving an accuracy of **89.74%**, with a sensitivity of **94.64%** and a precision of **92.99%**. It outperformed other models in detecting myocardial infarction.

## Summary of Findings

- Both machine learning and deep learning models were effective in classifying heart diseases using ECG signals.
- The **ConvNetQuacke** deep learning model showed the best overall performance.
- The dataset, after preprocessing, provided robust insights for model training, especially with the clear distinction between myocardial infarction and healthy controls.


## Instructions to Run the Project

1. **Dependencies**:
   - Libraries: `scikit-learn`, `TensorFlow`, `Keras`, `NumPy`, `Matplotlib`, `SciPy`

2. **Steps**:
   - Clone this repository.
   - Preprocess the ECG data using the provided preprocessing scripts.
   - Train the machine learning and deep learning models using the preprocessed data.
   - Evaluate the models using the provided evaluation metrics.

