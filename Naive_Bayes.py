import os
import numpy as np  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore

class NaiveBayes:
    def fit(self, X, y, alpha=1e-6):
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)

        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self._var = np.zeros((n_classes, n_features), dtype=np.float64)
        self._priors = np.zeros(n_classes, dtype=np.float64)

        for idx, c in enumerate(self._classes):
            X_c = X[y == c]
            self._mean[idx, :] = X_c.mean(axis=0)
            self._var[idx, :] = X_c.var(axis=0) + alpha  
            self._priors[idx] = X_c.shape[0] / float(n_samples)

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        posteriors = []

        for idx, c in enumerate(self._classes):
            prior = np.log(self._priors[idx])
            posterior = np.sum(np.log(self._pdf(idx, x) + 1e-10))  
            posterior += prior
            posteriors.append(posterior)

        return self._classes[np.argmax(posteriors)]

    def _pdf(self, class_idx, x):
        mean = self._mean[class_idx]
        var = self._var[class_idx]
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        pdf = numerator / denominator
        pdf[pdf < 1e-10] = 1e-10  
        return pdf

def load_and_scale_data(file_path):
    data = np.loadtxt(file_path)
    data[data == 1.00000000000000e+99] = np.nan 
    nan_indices = np.isnan(data)
    col_means = np.nanmean(data, axis=0)
    data[nan_indices] = np.take(col_means, np.where(nan_indices)[1])
    
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    
    return data

def calculate_accuracy(y_true, y_pred):
    correct = np.sum(y_true == y_pred)
    total = len(y_true)
    return correct / total

if __name__ == "__main__":
    data_dir = "data"
    output_folder = "output"
    os.makedirs(output_folder, exist_ok=True)

    print(f"Processing dataset 1...")

    train_data_file = os.path.join(data_dir, "TrainData1.txt")
    train_label_file = os.path.join(data_dir, "TrainLabel1.txt")
    test_data_file = os.path.join(data_dir, "TestData1.txt")

    train_data = load_and_scale_data(train_data_file)
    train_labels = np.loadtxt(train_label_file).astype(int)  
    test_data = load_and_scale_data(test_data_file)

    num_labels = train_labels.shape[1] if len(train_labels.shape) > 1 else 1
    predictions = []
    label_accuracies = []

    for label_idx in range(num_labels):
        label_train = train_labels[:, label_idx] if num_labels > 1 else train_labels

        nb = NaiveBayes()
        nb.fit(train_data, label_train)

        train_predictions = nb.predict(train_data)
        label_accuracy = calculate_accuracy(label_train, train_predictions)
        label_accuracies.append(label_accuracy)

        test_predictions = nb.predict(test_data)
        predictions.append(test_predictions)

    overall_accuracy = np.mean(label_accuracies)
    print(f"Naive Bayes Overall Classification Accuracy (on training data for dataset 1): {overall_accuracy:.4f}")

    all_predictions = np.array(predictions).T  
    combined_output_file = os.path.join(output_folder, "JenksKhundmiriClassification1.txt")
    np.savetxt(combined_output_file, all_predictions, fmt="%d", delimiter="\t")
    print(f"Predictions for dataset 1 saved to {combined_output_file}")