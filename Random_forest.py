import os
import numpy as np # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore
from sklearn.ensemble import RandomForestClassifier # type: ignore
from sklearn.model_selection import train_test_split, cross_val_score # type: ignore
from sklearn.metrics import accuracy_score # type: ignore

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
    return accuracy_score(y_true, y_pred)

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

    print("Processing dataset 2...")

    train_data_file = os.path.join(data_dir, "TrainData2.txt")
    train_label_file = os.path.join(data_dir, "TrainLabel2.txt")
    test_data_file = os.path.join(data_dir, "TestData2.txt")

    train_data = load_and_scale_data(train_data_file)
    train_labels = np.loadtxt(train_label_file).astype(int) 
    test_data = load_and_scale_data(test_data_file)

    num_labels = train_labels.shape[1] if len(train_labels.shape) > 1 else 1
    predictions = []
    label_accuracies = []

    for label_idx in range(num_labels):
        label_train = train_labels[:, label_idx] if num_labels > 1 else train_labels

        X_train, X_val, y_train, y_val = train_test_split(train_data, label_train, test_size=0.2, random_state=42)

        rf = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=5, random_state=42)
        rf.fit(X_train, y_train)

        val_predictions = rf.predict(X_val)
        val_accuracy = calculate_accuracy(y_val, val_predictions)
        label_accuracies.append(val_accuracy)

        test_predictions = rf.predict(test_data)
        predictions.append(test_predictions)

    overall_accuracy = np.mean(label_accuracies)
    print(f"Random Forest Overall Validation Accuracy (on validation data for dataset 2): {overall_accuracy:.4f}")

    all_predictions = np.array(predictions).T  
    combined_output_file = os.path.join(output_folder, "JenksKhundmiriClassification2.txt")
    np.savetxt(combined_output_file, all_predictions, fmt="%d", delimiter="\t")
    print(f"Predictions for dataset 2 saved to {combined_output_file}")