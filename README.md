# AA1 Project: Voter turnout prediction
Authors:
- Benet Ramió
- Pau Amargant
  
#
This is a machine learning project that aims to predict whether an eligible voter voted in the 2020 US presidential election. The project covers the entire cycle of a machine learning project, from data preprocessing to model evaluation.

# Usage
## Required Packages
A Python 3.11 [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html) environment on Linux was used for development and testing. The project may not work on other versions of Python or operating. The conda environment can be created using the following command:
```bash
conda create --name AA1_env python=3.11
conda activate AA1_env
```

 This project requires the following packages to be installed.
- numpy
- pandas
- scikit-learn
- xgboost
- catboost
- optuna
- imbalanced-learn
- category-encoders
- matplotlib
- plotly

The full list of packages can be found in the requirements.txt file. The packages can be installed using the following command:
```bash
pip install -r requirements.txt
```
## Dataset
The dataset is included in the submission. The codebook includes description of the variables and how they are coded.
## Replicating the results
The results can be replicated by running the Jupyter notebooks included in the submission. For the sake of reproducibility, the notebooks have been run using the seed 42.
The `pipeline.py` file includes functions and classes that are used in the notebooks.
The search for the optimal hyperparameters can be found in the Parameter_selection folder. The hyperparameter search notebooks have a long running time and can be skipped.
The exploration notebook plots interesting histograms and graphs of the data.
The `modeling_bests` notebook fits and compares each model using the best hyperparameters found
The `final_evaluation` notebook fits de model on the whole train split and evaluates its performance with the test split.

