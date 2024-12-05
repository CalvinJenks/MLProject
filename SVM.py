import os
import numpy as np # type: ignore
from sklearn.svm import SVC # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore

class SupportVectorMachine:
    def __init__(self, kernel='linear', C=1.0):
        self.model = SVC(kernel=kernel, C=C, probability=True)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

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

if __name__ == "__main__":
    data_dir = "data"
    output_folder = "output"
    os.makedirs(output_folder, exist_ok=True)

    print(f"Processing dataset 6...")

    train_data_file = os.path.join(data_dir, "TrainData6.txt")
    train_label_file = os.path.join(data_dir, "TrainLabel6.txt")
    test_data_file = os.path.join(data_dir, "TestData6.txt")

    train_data = load_and_scale_data(train_data_file)
    train_labels = np.loadtxt(train_label_file).astype(int)  
    test_data = load_and_scale_data(test_data_file)

    num_labels = train_labels.shape[1] if len(train_labels.shape) > 1 else 1
    predictions = []
    label_accuracies = []

    for label_idx in range(num_labels):
        label_train = train_labels[:, label_idx] if num_labels > 1 else train_labels

        svm = SupportVectorMachine(kernel='linear', C=1.0)
        svm.fit(train_data, label_train)

        train_predictions = svm.predict(train_data)
        label_accuracy = calculate_accuracy(label_train, train_predictions)
        label_accuracies.append(label_accuracy)

        test_predictions = svm.predict(test_data)
        predictions.append(test_predictions)

    overall_accuracy = np.mean(label_accuracies)
    print(f"SVM Overall Classification Accuracy (on training data for dataset 6): {overall_accuracy:.4f}")

    all_predictions = np.array(predictions).T 
    combined_output_file = os.path.join(output_folder, "JenksKhundmiriClassification6.txt")
    np.savetxt(combined_output_file, all_predictions, fmt="%d", delimiter="\t")
    print(f"Predictions for dataset 6 saved to {combined_output_file}")