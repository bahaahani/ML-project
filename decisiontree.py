import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split, KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

def load_images_from_folder(folder, label):
    images = []
    labels = []
    for filename in os.listdir(folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):  # Check if it's an image file
            img = Image.open(os.path.join(folder, filename)).convert('L')  # Convert to grayscale
            img = img.resize((64, 64))  # Resize the image
            if img is not None:
                images.append(np.array(img).flatten())
                labels.append(label)
    return images, labels

# Load all images and their labels
folder_path = './natural_images'
classes = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
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

# Data Normalization
scaler = StandardScaler()
data = scaler.fit_transform(data)

# Randomly select subset of features
n_features = 100  # Example: 100 features
feature_indices = np.random.choice(data.shape[1], n_features, replace=False)
data = data[:, feature_indices]

# Reserve the last 15 records for future prediction
X_future = data[-15:]
y_future = labels[-15:]
X = data[:-15]
y = labels[:-15]

# Split the data into training and testing sets (70% training, 30% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Decision Tree Classifier
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# K-fold Cross Validation
kf = KFold(n_splits=5)
acc_scores = []
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc_scores.append(accuracy_score(y_test, y_pred))

# Average accuracy from K-fold cross-validation
avg_accuracy = np.mean(acc_scores)
print(f"Average Accuracy from K-fold CV: {avg_accuracy}")

# Evaluate on the test set
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on Test Set: {accuracy}")
