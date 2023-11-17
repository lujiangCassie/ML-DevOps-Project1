# library doc string
'''
This code contains analysis and plots of churn customer 

Author : Lu Jiang

Date : 17th November 2023
'''


# import libraries
import os
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_roc_curve, classification_report
os.environ['QT_QPA_PLATFORM'] = 'offscreen'


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    df = pd.read_csv(pth)

    return df


def perform_eda(df):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''
    eda_df = df.copy(deep=True) 

    eda_df['Churn'] = eda_df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    plt.figure(figsize=(20, 10))
    eda_df['Churn'].hist()
    plt.savefig(fname='./images/eda/churn_distribution.png')

    plt.figure(figsize=(20, 10))
    eda_df['Customer_Age'].hist()
    plt.savefig(fname='./images/eda/customer_age_distribution.png')

    plt.figure(figsize=(20, 10))
    eda_df.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.savefig(fname='./images/eda/marital_status_distribution.png')

    plt.figure(figsize=(20, 10))
    sns.histplot(eda_df['Total_Trans_Ct'], kde=True)
    plt.savefig(fname='./images/eda/total_transaction_distribution.png')

    plt.figure(figsize=(20, 10))
    sns.heatmap(eda_df.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.savefig(fname='./images/eda/heatmap.png')

    return eda_df


def encoder_helper(df, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''
    encoder_df = df.copy(deep=True)
    
    for category in category_lst:
        column_lst = []
        column_groups = df.groupby(category).mean()['Churn']

        for val in df[category]:
            column_lst.append(column_groups.loc[val])

        if response:
            encoder_df[category + '_' + response] = column_lst
        else:
            encoder_df[category] = column_lst
   
    return encoder_df


def perform_feature_engineering(df, response):
    '''
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    cat_columns = [ 'Gender', 'Education_Level', 'Marital_Status','Income_Category', 'Card_Category'  ]

    encoded_df = encoder_helper(df, category_lst=cat_columns, response=response)

    y = encoded_df['Churn']     

    X = pd.DataFrame()         

    keep_cols = ['Customer_Age', 'Dependent_count', 'Months_on_book',
                 'Total_Relationship_Count', 'Months_Inactive_12_mon',
                 'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
                 'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
                 'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
                 'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn',
                 'Income_Category_Churn', 'Card_Category_Churn']

    X[keep_cols] = encoded_df[keep_cols]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    plt.rc('figure', figsize=(6, 6))
    plt.text(0.01, 1.25,
             str('Random Forest Train'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05,
             str(classification_report(y_test, y_test_preds_rf)),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.6,
             str('Random Forest Test'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7,
             str(classification_report(y_train, y_train_preds_rf)),
             {'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    plt.savefig(fname='./images/results/rf_results.png')

    plt.rc('figure', figsize=(6, 6))
    plt.text(0.01, 1.25,
             str('Logistic Regression Train'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05,
             str(classification_report(y_train, y_train_preds_lr)),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.6,
             str('Logistic Regression Test'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7,
             str(classification_report(y_test, y_test_preds_lr)),
             {'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    plt.savefig(fname='./images/results/logistic_results.png')


def feature_importance_plot(model, features, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            features: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    importances = model.best_estimator_.feature_importances_
    indices = np.argsort(importances)[::-1]

    names = [features.columns[i] for i in indices]

    plt.figure(figsize=(25, 15))

    plt.title("Feature Importance")
    plt.ylabel('Importance')
    plt.bar(range(features.shape[1]), importances[indices])
    plt.xticks(range(features.shape[1]), names, rotation=90)
    plt.savefig(fname=output_pth + 'feature_importances.png')


def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    rfc = RandomForestClassifier(random_state=2)
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)
    lrc.fit(X_train, y_train)

    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')

    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    plt.figure(figsize=(15, 8))
    axis = plt.gca()
    plot_roc_curve(lrc, X_test, y_test, ax=axis, alpha=0.8)
    plot_roc_curve(cv_rfc.best_estimator_, X_test, y_test, ax=axis, alpha=0.8)
    plt.savefig(fname='./images/results/roc_curve_result.png')

    classification_report_image(y_train, y_test,
                                y_train_preds_lr, y_train_preds_rf,
                                y_test_preds_lr, y_test_preds_rf)

    feature_importance_plot(model=cv_rfc,
                            features=X_test,
                            output_pth='./images/results/')


if __name__ == '__main__':

    bank_df = import_data(pth='./data/bank_data.csv')

    EDA_df = perform_eda(bank_df)
    
    X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = perform_feature_engineering(EDA_df, response='Churn')

    train_models(X_train=X_TRAIN,
                 X_test=X_TEST,
                 y_train=Y_TRAIN,
                 y_test=Y_TEST)
    
