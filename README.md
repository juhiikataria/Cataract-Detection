# Cataract Detection

This project aims to detect cataracts using various deep learning models, including RNN, CNN, and VGG16, and provides a user-friendly interface for easy image classification.

## Notebooks Overview

1. **Cataract_RNN.ipynb**
   - **Purpose:** Implements a Recurrent Neural Network (RNN) for cataract detection.
   - **Key Features:**
     - Data loading and preprocessing from the Merged Dataset.
     - Utilizes TensorFlow and Keras for building and training the RNN model.
   - **Dataset:** Utilizes a dataset stored on Google Drive, split into training, validation, and test sets.

2. **Cataract_CNN_1.ipynb**
   - **Purpose:** Implements a Convolutional Neural Network (CNN) for cataract detection.
   - **Key Features:**
     - Data loading and preprocessing.
     - Uses TensorFlow and Keras to construct a CNN model.
   - **Dataset:** Uses the same Merged Dataset with standard preprocessing techniques.

3. **cataract_detect_vgg_TL.ipynb**
   - **Purpose:** Utilizes Transfer Learning with VGG16 for cataract detection.
   - **Key Features:**
     - Applies data augmentation techniques to enhance model performance.
     - Constructs a model using VGG16 architecture for improved accuracy.
     - Implements transfer learning to leverage pre-trained weights.
   - **Dataset:** Uses the Merged Dataset, employing extensive augmentation techniques for better results.

4. **interface.ipynb**
   - **Purpose:** Provides a web interface for cataract detection using trained models.
   - **Key Features:**
     - Uses Gradio to create a simple web interface for image upload and prediction.
     - Supports multiple models (CNN, RNN, VGG16) and provides the most confident prediction.
   - **Models:** Loads pre-trained models for making predictions on uploaded images.

## How to Use

1. **Data Preparation:**
   - Ensure the dataset is structured as follows:
     ```
     Merged Dataset/
     ├── Train/
     ├── Val/
     └── Test/
     ```
   - Update paths in each notebook to point to your dataset location.

2. **Model Training:**
   - Run each notebook sequentially for model training. Start with `Cataract_RNN.ipynb`, then `Cataract_CNN_1.ipynb`, followed by `cataract_detect_vgg_TL.ipynb`.
   - Each notebook contains code for data preprocessing, model construction, and training.

3. **Interface:**
   - Use `interface.ipynb` to launch a web interface.
   - Ensure that the paths to the trained model files are correctly set in the notebook.
   - Upload an image to detect cataracts using the interface.

## Requirements

- Python 3.x
- TensorFlow
- Keras
- Gradio
- Google Colab for notebook execution
