# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
In this project, we aim to predict credit card customers that are most likely to churn according to customers credit card. The project contains modular functions which follows coding PEP8 using pylint. It also includes the scripts for test on every unit of the code.

## Files and data description
Folders:
    Data
        bank data csv file
    images 
        eda: --> distribution of multiple key features and heatmap of feature correlations
        results: --> result plots of model and roc curve
    models 
        Saved models in .pkl format
    logs
        test logging file

project files

    churn_library.py  --> refactoring code
    churn_notebook.ipnyb  --> original code
    test_churn_script_logging_and_tests.py --> test code

text file
    requirements.txt  --> python libraries
    readme.md  --> guidence file


## Running Files
You can run the model with command:

python churn_library.py

If every module in code runs without problems, the best fitting model will be saved in models folder. eda images and results metrics will be saved in images/eda and images/results folders.

To test the code, you can run:

python churn_script_logging_and_tests.py

All logs from unit test will be saved in churn_library.log file inside logs folder.



