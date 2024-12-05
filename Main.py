import subprocess

scripts = [
    "Naive_bayes.py",
    "Random_Forest.py",
    "LDA.py",
    "Logistic_Regression.py",
    "KNN.py",
    "SVM.py"
]

def run_scripts(script_list):
    for script in script_list:
        try:
            result = subprocess.run(["python", script], capture_output=True, text=True, check=True)
            print(result.stdout)  
        except subprocess.CalledProcessError as e:
            print(e.stderr) 

if __name__ == "__main__":
    run_scripts(scripts)
