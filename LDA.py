import os
import numpy as np # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis # type: ignore

def load_data(file_path, delimiter=None):
    if delimiter is None:
        with open(file_path, 'r') as file:
            first_line = file.readline()
            if ',' in first_line:
                delimiter = ','
            elif '\t' in first_line:
                delimiter = '\t'
            else:
                raise ValueError(f"Unable to determine delimiter for file: {file_path}")
    try:
        data = np.loadtxt(file_path, delimiter=delimiter)
        data[data == 1.00000000000000e+99] = np.nan 
        return data
    except ValueError as e:
        raise ValueError(f"Error loading data from {file_path}: {e}")

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
    data = load_data(file_path)
    data = handle_missing_values(data)
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    return data

def apply_log_transformation(data):
    data[data <= 0] = 1e-10
    return np.log1p(data)

if __name__ == "__main__":
    train_data_file = "data/TrainData3.txt"
    train_label_file = "data/TrainLabel3.txt"
    test_data_file = "data/TestData3.txt"
    output_folder = "output"
    os.makedirs(output_folder, exist_ok=True)

    try:
        print("Processing dataset 3...")

        train_data = load_and_scale_data(train_data_file)
        train_labels = np.loadtxt(train_label_file, dtype=int)  
        test_data = load_and_scale_data(test_data_file)

        num_labels = train_labels.shape[1] if len(train_labels.shape) > 1 else 1
        predictions = []
        label_accuracies = []

        for label_idx in range(num_labels):
            label_train = train_labels[:, label_idx] if num_labels > 1 else train_labels

            lda = LinearDiscriminantAnalysis()
            lda.fit(train_data, label_train)

            train_predictions = lda.predict(train_data)
            label_accuracy = calculate_accuracy(label_train, train_predictions)
            label_accuracies.append(label_accuracy)

            test_predictions = lda.predict(test_data)
            predictions.append(test_predictions)

        overall_accuracy = np.mean(label_accuracies)
        print(f"LDA Overall Classification Accuracy (on training data for dataset 3): {overall_accuracy:.4f}")

        all_predictions = np.array(predictions).T  
        combined_output_file = os.path.join(output_folder, "JenksKhundmiriClassification3.txt")
        np.savetxt(combined_output_file, all_predictions, fmt="%d", delimiter="\t")
        print(f"Predictions for dataset 3 saved to: {combined_output_file}")

    except FileNotFoundError as e:
        print(f"FileNotFoundError: {e}")
    except ValueError as e:
        print(f"ValueError: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")