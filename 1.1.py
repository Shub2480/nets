# inference.py
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Load the saved model
model = load_model("mnist_cnn_model.h5")

def make_inference(model, input_image):
    """
    Perform inference on a single image.
    :param model: Trained model
    :param input_image: Preprocessed image of shape (28, 28)
    :return: Predicted label
    """
    # Reshape and normalize the image
    input_image = input_image.reshape(1, 28, 28, 1) / 255.0
    
    # Make prediction
    predictions = model.predict(input_image)
    predicted_label = np.argmax(predictions)
    
    return predicted_label

# Test the inference function
# Here we simulate loading some test data
from tensorflow.keras.datasets import mnist
(_, _), (x_test, y_test) = mnist.load_data()

# Normalize test data
x_test = x_test / 255.0
x_test = x_test.reshape(-1, 28, 28, 1)

# Select an image for testing
test_image = x_test[0]  # First image from the test set
true_label = y_test[0]  # True label for the selected image

# Predict the label
predicted_label = make_inference(model, test_image)

# Display the image and prediction
plt.imshow(test_image, cmap='gray')
plt.title(f"True Label: {true_label}, Predicted: {predicted_label}")
plt.axis('off')
plt.show()
