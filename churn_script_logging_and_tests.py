'''
This code contains all kinds of test on churn customer prediction including dataframe and analyzed model

Author : Lu Jiang

Date 20th December 2023
'''
import os
import logging
from math import ceil
import churn_library as clib

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import():
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        df = clib.import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda():
    '''
    test perform eda function
    '''
    df = clib.import_data("./data/bank_data.csv")
    try:
        clib.perform_eda(df)
        logging.info("Testing perform_eda: SUCCESS")
    except KeyError as err:              
        logging.error('Column "%s" not found', err.args[0])
        raise err
            
    try:
        assert os.path.isfile('./images/eda/churn_distribution.png')
        assert os.path.isfile('./images/eda/customer_age_distribution.png')
        assert os.path.isfile('./images/eda/heatmap.png')
        assert os.path.isfile('./images/eda/marital_status_distribution.png')
        assert os.path.isfile('./images/eda/total_transaction_distribution.png')
        logging.info("Testing perform_eda: SUCCESS")

    except AssertionError as err:
        logging.error("Testing perform_eda: error on checking images files.")
        raise err

def test_encoder_helper():
    '''
    test encoder helper
    '''
    df = clib.import_data("./data/bank_data.csv")
    df['Churn'] = df['Attrition_Flag'].apply(lambda val: 0 if val=="Existing Customer" else 1)  
    cat_columns = ['Gender', 'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category']

    try:
        encoded_df = clib.encoder_helper(df, category_lst=[], response=None)

        assert encoded_df.equals(df) is True
        logging.info("Testing encoder_helper(data_frame, category_lst=[]): SUCCESS")
    except AssertionError as err:
        logging.error("Testing encoder_helper(data_frame, category_lst=[]): ERROR")
        raise err
          
    try:
        encoded_df = clib.encoder_helper(df, category_lst=cat_columns, response=None)
        assert encoded_df.columns.equals(df.columns) is True
        assert encoded_df.equals(df) is False
        logging.info("Testing encoder_helper(data_frame, category_lst=cat_columns, response=None): SUCCESS")
    except AssertionError as err:
        logging.error("Testing encoder_helper(data_frame, category_lst=cat_columns, response=None): ERROR")
        raise err
                
    try:
        encoded_df = clib.encoder_helper(df, category_lst=cat_columns, response='Churn')
        
        assert encoded_df.columns.equals(df.columns) is False   
        assert encoded_df.equals(df) is False
        assert len(encoded_df.columns) == len(df.columns) + len(cat_columns)    
        logging.info("Testing encoder_helper(data_frame, category_lst=cat_columns, response='Churn'): SUCCESS")
    except AssertionError as err:
        logging.error("Testing encoder_helper(data_frame, category_lst=cat_columns, response='Churn'): ERROR")
        raise err

def test_perform_feature_engineering():
    '''
    test perform_feature_engineering
    '''
    
    df = clib.import_data("./data/bank_data.csv")

    df['Churn'] = df['Attrition_Flag'].apply(lambda val: 0 if val=="Existing Customer" else 1)

    try:
        (_, X_test, _, _) = clib.perform_feature_engineering(df, response='Churn')

        assert 'Churn' in df.columns
        logging.info("Testing perform_feature_engineering. `Churn` column is present: SUCCESS")
    except KeyError as err:
        logging.error('The `Churn` column is not present in the DataFrame: ERROR')
        raise err

    try:
        assert (X_test.shape[0] == ceil(df.shape[0]*0.3)) is True   
        logging.info('Testing perform_feature_engineering. DataFrame sizes are consistent: SUCCESS')
    except AssertionError as err:
        logging.error('Testing perform_feature_engineering. DataFrame sizes are not correct: ERROR')
        raise err

def test_train_models():
    '''
    test train_models
    '''
    
    df = clib.import_data("./data/bank_data.csv")

    df['Churn'] = df['Attrition_Flag'].apply(lambda val: 0 if val=="Existing Customer" else 1)

    (X_train, X_test, y_train, y_test) = clib.perform_feature_engineering(df, response='Churn')

    try:
        clib.train_models(X_train, X_test, y_train, y_test)
        assert os.path.isfile("./models/logistic_model.pkl") is True
        logging.info('File %s was found', 'logistic_model.pkl')
    except AssertionError as err:
        logging.error('Not such file on disk')
        raise err

    try:
        assert os.path.isfile("./models/rfc_model.pkl") is True
        logging.info('File %s was found', 'rfc_model.pkl')
    except AssertionError as err:
        logging.error('Not such file on disk')
        raise err

    try:
        assert os.path.isfile('./images/results/roc_curve_result.png') is True
        logging.info('File %s was found', 'roc_curve_result.png')
    except AssertionError as err:
        logging.error('Not such file on disk')
        raise err

    try:
        assert os.path.isfile('./images/results/rf_results.png') is True
        logging.info('File %s was found', 'rf_results.png')
    except AssertionError as err:
        logging.error('Not such file on disk')
        raise err

    try:
        assert os.path.isfile('./images/results/logistic_results.png') is True
        logging.info('File %s was found', 'logistic_results.png')
    except AssertionError as err:
        logging.error('Not such file on disk')
        raise err

    try:
        assert os.path.isfile('./images/results/feature_importances.png') is True
        logging.info('File %s was found', 'feature_importances.png')
    except AssertionError as err:
        logging.error('Not such file on disk')
        raise err


if __name__ == "__main__":
    test_import()
    test_eda()
    test_encoder_helper()
    test_perform_feature_engineering()
    test_train_models()








