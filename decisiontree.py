import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


def load_images_from_folder(folder, label):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img = Image.open(os.path.join(folder, filename)).convert(
            'L')  # Convert to grayscale
        img = img.resize((64, 64))  # Resize the image
        if img is not None:
            images.append(np.array(img).flatten())
            labels.append(label)
    return images, labels


# Load all images and their labels
folder_path = './natural_images'
classes = os.listdir(folder_path)
data = []
labels = []
for idx, class_name in enumerate(classes):
    class_path = os.path.join(folder_path, class_name)
    images, image_labels = load_images_from_folder(class_path, idx)
    data.extend(images)
    labels.extend(image_labels)

# Convert lists to numpy arrays
data = np.array(data)
labels = np.array(labels)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=42)

# Train Decision Tree Classifier
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Predict and Evaluate
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
