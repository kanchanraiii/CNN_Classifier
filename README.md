# CNN Classifier - Cats vs. Dogs

## Dataset  
- **Source:** `tf.keras.datasets` (Built-in TensorFlow dataset)  
- **Loading the Data:**  
  ```python
  import tensorflow as tf

  (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

- **Preprocessing:**  
  - Resized images to (150, 150)  
  - Normalized pixel values to [0,1]  

## Model Architecture  
- **Type:** Convolutional Neural Network (CNN)  
- **Layers:**  
  - `Conv2D` → `ReLU` → `MaxPooling`  
  - `Conv2D` → `ReLU` → `MaxPooling`  
  - `Flatten` → `Dense` → `Dropout`  
  - Output layer (`Sigmoid` for classification)  

## Required Libraries  
```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
```
## Model Training  
The model is trained using the Adam optimizer and sparse categorical cross-entropy loss. Training is done for 5 epochs with validation data.  

## Model Saving  
The trained model is saved in HDF5 format (`.h5`) to allow easy loading and inference later.  

## Model Loading  
The saved model can be reloaded without recompilation to make predictions on new images.  
