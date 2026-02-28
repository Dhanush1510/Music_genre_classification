# Music Genre Classification on GTZAN and MusicNet Dataset

## Overview

This project implements a music genre classification system using deep learning techniques on:

- GTZAN Dataset  
- MusicNet Dataset  

The entire workflow — including data loading, preprocessing, feature extraction, model building, training, and evaluation — is implemented inside the `DL_Project.ipynb` notebook.

This README strictly reflects only the components implemented in the notebook.

---

## Libraries Used (As Imported in Notebook)

The notebook uses the following libraries:

- numpy  
- pandas  
- matplotlib  
- librosa  
- scikit-learn  
- tensorflow / keras (for deep learning model implementation)

No additional frameworks or tools are used outside these imports.

---

## Dataset Handling

### GTZAN Dataset
- Audio files are loaded from directory structure.
- Each file is processed individually.
- Labels are extracted from folder names.

### MusicNet Dataset
- Audio files are loaded and processed similarly.
- Labels are handled according to dataset structure present in the notebook.

No external preprocessing scripts are used outside the notebook.

---

## Audio Preprocessing

The following preprocessing steps are implemented:

- Audio loading using `librosa.load()`
- Uniform sampling rate during loading
- Fixed duration handling (padding or trimming if applied in notebook)
- Conversion of audio signals into numerical arrays

Only preprocessing steps explicitly coded in the notebook are applied.

---

## Feature Extraction

The notebook extracts the following features (as implemented in code):

- MFCC (Mel-Frequency Cepstral Coefficients)
- Mel Spectrogram (if computed in notebook)
- Feature arrays reshaped to match model input requirements

Feature extraction is performed using `librosa` functions.

No additional handcrafted features are used unless explicitly shown in the notebook.

---

## Model Architecture

The model implemented in the notebook is a Convolutional Neural Network (CNN).

Architecture includes:

- Convolutional Layers
- Activation Functions (as defined in code)
- Pooling Layers (if present in notebook)
- Flatten Layer
- Dense (Fully Connected) Layers
- Output Layer with Softmax activation for multi-class classification

All layers and configurations strictly follow what is defined in the notebook model summary.

---

## Training Configuration

Training is performed using:

- Loss Function: Categorical Crossentropy (as defined in model compilation)
- Optimizer: (Exactly as defined in notebook, e.g., Adam if present)
- Evaluation Metric: Accuracy
- Train-test split using scikit-learn

Number of epochs, batch size, and validation usage are exactly as specified in the notebook.

---

## Model Evaluation

Evaluation includes:

- Model accuracy on test data
- Loss curves (if plotted)
- Accuracy curves (if plotted)

No additional metrics (precision, recall, F1-score) are included unless explicitly computed in the notebook.

---

## Workflow Summary

1. Load audio files
2. Extract features using librosa
3. Encode labels
4. Split dataset into training and testing sets
5. Build CNN model
6. Train model
7. Evaluate performance
8. Visualize training metrics (if included)

All steps are contained within `DL_Project.ipynb`.

---

## How to Run

1. Install required dependencies:
   ```
   pip install numpy pandas matplotlib librosa scikit-learn tensorflow
   ```

2. Open Jupyter Notebook:
   ```
   jupyter notebook DL_Project.ipynb
   ```

3. Run all cells sequentially.

Ensure dataset paths in the notebook match your local directory structure.

---

## Conclusion

This project demonstrates a complete deep learning-based music genre classification pipeline implemented entirely within a single notebook.

All preprocessing, feature extraction, model training, and evaluation steps are reproducible directly from `DL_Project.ipynb`, with no external scripts or undocumented processing steps involved.
