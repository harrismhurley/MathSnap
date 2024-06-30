import os  # Importing the os module to interact with the operating system
import cv2  # Importing OpenCV for image processing
import numpy as np  # Importing NumPy for numerical operations
import matplotlib.pyplot as plt  # Importing Matplotlib for plotting
import tensorflow as tf  # Importing TensorFlow for building and training models

# The following lines are commented out. They demonstrate how to load, preprocess, build, compile, and train a model on the MNIST dataset.

# Load the MNIST dataset
# mnist = tf.keras.datasets.mnist

# Split the dataset into training and test sets
# (x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the training and test data
# x_train = tf.keras.utils.normalize(x_train, axis=1)
# x_test = tf.keras.utils.normalize(x_test, axis=1)

# Create a sequential model
# model = tf.keras.models.Sequential()

# Add a flatten layer to convert each 28x28 image into a 784-element 1D array
# model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))

# Add a dense layer with 128 neurons and ReLU activation function
# model.add(tf.keras.layers.Dense(128, activation='relu'))

# Add another dense layer with 128 neurons and ReLU activation function
# model.add(tf.keras.layers.Dense(128, activation='relu'))

# Add an output layer with 10 neurons and softmax activation function (for classification into 10 classes)
# model.add(tf.keras.layers.Dense(10, activation='softmax'))

# Compile the model with Adam optimizer and sparse categorical crossentropy loss function
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model on the training data for 3 epochs
# model.fit(x_train, y_train, epochs=3)

# Save the trained model to a file
# model.save('digits_model.keras')

# Load the pre-trained model from the file
model = tf.keras.models.load_model('digits_model.keras')

# The following lines are commented out. They demonstrate how to evaluate the model on the test data.

# Evaluate the model on the test data
# loss, accuracy = model.evaluate(x_test, y_test)

# Print the loss and accuracy
# print(loss)
# print(accuracy)

# Initialize the image number
image_number = 1

# Loop through the images in the digits directory
while os.path.isfile(f"digits/digit{image_number}.png"):
    try:
        # Read the image in grayscale mode
        img = cv2.imread(f"digits/digit{image_number}.png")[:,:,0]
        
        # Invert the image colors (black becomes white, and vice versa)
        img = np.invert(np.array([img]))
        
        # Make a prediction using the model
        prediction = model.predict(img)
        
        # Print the predicted digit
        print(f"this digit is probably a {np.argmax(prediction)}")
        
        # Display the image
        plt.imshow(img[0], cmap = plt.cm.binary)
        plt.show()
    except:
        # Print an error message if something goes wrong
        print("Error!")
    finally:
        # Increment the image number
        image_number += 1

# Final accuracy note
# accuracy 5/12
