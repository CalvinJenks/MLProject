MACHINE LEARNING PROJECT

This project contains multiple machine learning model scripts and a main script (main.py) that runs them sequentially. You can use this to run various models like Naive Bayes, Random Forest, SVM, etc., on your datasets. The script also handles missing value estimation.

REQUIREMENTS

To run the project, you'll need to have the following Python libraries installed:

numpy
scikit-learn

You can install these libraries using pip.

INSTALLATION

Download the repository:
Download or clone the repository to your local machine.

Install Dependencies:
Use pip to install the required libraries. Open a terminal and run:

pip install numpy scikit-learn

If you are using a virtual environment, ensure it's activated before running the command.
Ensure Python is Available:

Verify that you have Python 3 installed. You can check this by running:

python --version

If the command returns Python 3.x, you're good to go. If it returns Python 2.x, you may need to install Python 3 or adjust the python3 references in the code to python.

You can modify the main script (main.py) if needed to replace python3 with python in the subprocess call.
Running the Code

To run the project, follow these steps:

Place Your Data:
Ensure that your dataset files (e.g., TrainData1.txt, TrainLabel1.txt, etc.) are placed in the data directory. The code assumes that these files are in the same directory structure.

Modify main.py if necessary:
If your system uses python instead of python3, open the main.py script and modify the subprocess.run line:

result = subprocess.run(["python3", script], capture_output=True, text=True, check=True)
Change it to:

result = subprocess.run(["python", script], capture_output=True, text=True, check=True)

Run the Main Script:
In your terminal, navigate to the project directory where main.py is located. Run the following command:

python main.py
This will execute the main.py script, which in turn runs each of the machine learning model scripts in sequence.
Output

The results of each script will be printed in the terminal. Predictions will be saved to the output directory in files like JenksKhundmiriClassification1.txt.

Troubleshooting

Missing Libraries: If you encounter an error about missing libraries, make sure you've installed the required dependencies using pip install numpy scikit-learn.

Python Version: If you encounter issues related to Python versioning, ensure that Python 3.x is installed and adjust the subprocess.run command in main.py to use python instead of python3.