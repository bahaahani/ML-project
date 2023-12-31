import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score


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
classes = [d for d in os.listdir(folder_path) if os.path.isdir(
    os.path.join(folder_path, d))]
data = []
labels = []
for idx, class_name in enumerate(classes):
    class_path = os.path.join(folder_path, class_name)
    if os.path.isdir(class_path):  # Check if it's a directory
        images, image_labels = load_images_from_folder(class_path, idx)
        data.extend(images)
        labels.extend(image_labels)


# Convert lists to numpy arrays
data = np.array(data)
labels = np.array(labels)

# Data Normalization
scaler = StandardScaler()
data = scaler.fit_transform(data)

# Randomly select subset of features (for simplicity, we'll use a fixed number of features)
n_features = 100  # Example: 100 features
feature_indices = np.random.choice(data.shape[1], n_features, replace=False)
data = data[:, feature_indices]

# Reserve the last 15 records for future prediction
X_future = data[-15:]
y_future = labels[-15:]
X = data[:-15]
y = labels[:-15]

# Split the data into training and testing sets (70% training, 30% testing)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

# Train KNN Classifier
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

# K-fold Cross Validation with additional metrics
kf = KFold(n_splits=5)
acc_scores = []
precision_scores = []
recall_scores = []
f1_scores = []

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    acc_scores.append(accuracy_score(y_test, y_pred))
    precision_scores.append(precision_score(y_test, y_pred, average='macro'))
    recall_scores.append(recall_score(y_test, y_pred, average='macro'))
    f1_scores.append(f1_score(y_test, y_pred, average='macro'))

# Average metrics from K-fold cross-validation
avg_accuracy = np.mean(acc_scores)
avg_precision = np.mean(precision_scores)
avg_recall = np.mean(recall_scores)
avg_f1_score = np.mean(f1_scores)

print(f"Average Accuracy: {avg_accuracy}")
print(f"Average Precision: {avg_precision}")
print(f"Average Recall: {avg_recall}")
print(f"Average F1-Score: {avg_f1_score}")

# Evaluate on the test set (Optional, as K-fold CV already evaluates the model)
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on Test Set: {accuracy}")
