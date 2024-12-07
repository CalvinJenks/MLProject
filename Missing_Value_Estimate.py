import os
import numpy as np
from sklearn.impute import KNNImputer

def load_data(file_path):
    data = np.loadtxt(file_path)
    data[data == 1.00000000000000e+99] = np.nan
    return data

def knn_impute_missing_values(data, n_neighbors=5):
    print("Imputing missing values with KNN...")
    imputer = KNNImputer(n_neighbors=n_neighbors)
    return imputer.fit_transform(data)

def save_data(file_path, data):
    print(f"Saving processed data to {file_path} \n")
    np.savetxt(file_path, data, fmt="%.6f", delimiter="\t")

def process_datasets(data_dir, output_dir, last_name="JenksKhundmiri"):
    os.makedirs(output_dir, exist_ok=True)

    datasets = [
        {"name": "Missing Dataset1", "filename": "MissingData1.txt"},
        {"name": "Missing Dataset2", "filename": "MissingData2.txt"},
        {"name": "Missing Dataset3", "filename": "MissingData3.txt"},
    ]
    
    for dataset in datasets:
        input_path = os.path.join(data_dir, dataset["filename"])
        output_path = os.path.join(output_dir, f"{last_name}MissingResult{dataset['name'][-1]}.txt")

        print(f"Processing {dataset['name']}...")
        data = load_data(input_path)
        data_imputed = knn_impute_missing_values(data)
        save_data(output_path, data_imputed)

if __name__ == "__main__":
    data_dir = "data"
    output_dir = "output"

    process_datasets(data_dir, output_dir, last_name="JenksKhundmiri")