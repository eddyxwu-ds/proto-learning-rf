import pickle
import numpy as np

# Load the downloaded MNIST data
with open("mnist.pkl", "rb") as f:
    # Handle the encoding issue mentioned in the original code
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    data = u.load()

# The data structure is typically (train_data, valid_data, test_data)
# Each is a tuple of (images, labels)
train_data, valid_data, test_data = data

# Combine train and validation data as expected by the code
train_images = np.vstack([train_data[0], valid_data[0]])
train_labels = np.hstack([train_data[1], valid_data[1]])

# Create the format expected by the code: [(train_images, train_labels), (test_images, test_labels)]
formatted_data = [(train_images, train_labels), (test_data[0], test_data[1])]

# Save in the format expected by the code
with open("mnist.data", "wb") as f:
    pickle.dump(formatted_data, f)

print("MNIST data converted and saved as mnist.data")
print(f"Training samples: {len(train_images)}")
print(f"Test samples: {len(test_data[0])}")
print(f"Image shape: {train_images[0].shape}") 