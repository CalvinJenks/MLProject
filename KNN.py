import os
import numpy as np  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore
from sklearn.neighbors import KNeighborsClassifier  # type: ignore # Importing k-NN classifier

class kNearestNeighbors:
    def __init__(self, n_neighbors=3):
        self.n_neighbors = n_neighbors
        self.knn = KNeighborsClassifier(n_neighbors=self.n_neighbors)

    def fit(self, X, y):
        self.knn.fit(X, y)

    def predict(self, X):
        return self.knn.predict(X)

def load_data(file_path):
    data = np.loadtxt(file_path)
    data[data == 1.00000000000000e+99] = np.nan 
    return data

def handle_missing_values(data):
    nan_indices = np.isnan(data)
    col_means = np.nanmean(data, axis=0)
    data[nan_indices] = np.take(col_means, np.where(nan_indices)[1])
    return data

def calculate_accuracy(y_true, y_pred):
    correct = np.sum(y_true == y_pred)
    total = len(y_true)
    return correct / total

def load_and_scale_data(file_path):
    data = np.loadtxt(file_path)
    data[data == 1.00000000000000e+99] = np.nan
    nan_indices = np.isnan(data)
    col_means = np.nanmean(data, axis=0)
    data[nan_indices] = np.take(col_means, np.where(nan_indices)[1])
    
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    return data

def apply_log_transformation(data):
    data[data <= 0] = 1e-10
    return np.log1p(data)

if __name__ == "__main__":
    data_dir = "data"
    output_folder = "output"
    os.makedirs(output_folder, exist_ok=True)

    print(f"Processing dataset 5...")

    train_data_file = os.path.join(data_dir, "TrainData5.txt")
    train_label_file = os.path.join(data_dir, "TrainLabel5.txt")
    test_data_file = os.path.join(data_dir, "TestData5.txt")

    train_data = load_and_scale_data(train_data_file)
    train_labels = np.loadtxt(train_label_file).astype(int)  # Avoid deprecation warning
    test_data = load_and_scale_data(test_data_file)

    num_labels = train_labels.shape[1] if len(train_labels.shape) > 1 else 1
    predictions = []
    label_accuracies = []

    for label_idx in range(num_labels):
        label_train = train_labels[:, label_idx] if num_labels > 1 else train_labels

        knn = kNearestNeighbors(n_neighbors=3) 
        knn.fit(train_data, label_train)

        train_predictions = knn.predict(train_data)
        label_accuracy = calculate_accuracy(label_train, train_predictions)
        label_accuracies.append(label_accuracy)

        test_predictions = knn.predict(test_data)
        predictions.append(test_predictions)

    overall_accuracy = np.mean(label_accuracies)
    print(f"k-NN Overall Classification Accuracy (on training data for dataset 5): {overall_accuracy:.4f}")

    all_predictions = np.array(predictions).T 
    combined_output_file = os.path.join(output_folder, "JenksKhundmiriClassification5.txt")
    np.savetxt(combined_output_file, all_predictions, fmt="%d", delimiter="\t")
    print(f"Predictions for dataset 5 saved to {combined_output_file}")