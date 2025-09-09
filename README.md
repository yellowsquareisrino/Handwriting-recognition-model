# Handwriting-recognition-model
Handwriting recognition model trained with TensorFlow/Keras on MNIST
# Handwriting Recognition with CNNs

A basic deep learning project in which I trained a Convolutional Neural Network (CNN) using **TensorFlow/Keras** to identify handwritten digits in the **MNIST dataset**.

I created this project to have hands-on experience with computer vision using neural networks and with saving and reusing trained models.

---

## What's inside this repo?
- **`handwriting_model.keras`** → trained model (you can load it directly without training).  
- **`handwriting.ipynb`** → my notebook with data preprocessing, training process, and testing.  
- **`requirements.txt`** → list of libraries required to run the notebook.  

---

## How to use it
If you want to give it a try:

```python
import tensorflow as tf
import numpy as np

# Load the model
model = tf.keras.models.load_model("handwriting_model.keras")
```

# Example prediction (swap `my_image` for your preprocessed image)
digit = np.expand_dims(my_image, axis=0)
prediction = model.predict(digit)
print("Predicted digit:", prediction.argmax())
