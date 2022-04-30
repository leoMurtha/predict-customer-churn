"""
Module containing all tests for the helper script churn_library.py

Author: Leonardo Meireles
Data: 29/04/2022
"""


import os
import logging
import glob
import pytest
import joblib
import pandas as pd
import matplotlib
import churn_library as cl

# Non interactive mode
matplotlib.use('Agg')


logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s', force=True)


@pytest.fixture
def data():
    """pytest fixture for the mocked bank data"""
    data_df = pd.read_csv('mock/bank_data_sample.csv')
    return data_df


@pytest.fixture
def response():
    """pytest fixture for response variable"""
    return 'Churn'


@pytest.fixture
def category_lst():
    """pytest fixture for the categorical columns"""
    return cl.CAT_COLUMNS


@pytest.fixture
def train_test_data():
    """pytest fixture for the mocked train_test data"""
    return joblib.load('mock/train_test_data_mock.pkl').get('X_train'), \
        joblib.load('mock/train_test_data_mock.pkl').get('X_test'), \
        joblib.load('mock/train_test_data_mock.pkl').get('y_train'), \
        joblib.load('mock/train_test_data_mock.pkl').get('y_test')


@pytest.fixture
def model_evaluation_data():
    """pytest fixture for the mocked model evaluation data"""
    return joblib.load('mock/model_evaluation_data.pkl').get('y_train'), \
        joblib.load('mock/model_evaluation_data.pkl').get('y_test'), \
        joblib.load('mock/model_evaluation_data.pkl').get('y_train_preds_lr'), \
        joblib.load('mock/model_evaluation_data.pkl').get('y_train_preds_lr'), \
        joblib.load('mock/model_evaluation_data.pkl').get('y_test_preds_lr'), \
        joblib.load('mock/model_evaluation_data.pkl').get('y_test_preds_rf')


@pytest.fixture
def rfc_model():
    """pytest fixture for the mocked random forest classifier model"""
    return joblib.load('./mock/rfc_model.pkl')


def test_import():
    '''
    test data import
    '''
    try:
        data_df = cl.import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_data: The file wasn't found")
        raise err

    try:
        assert data_df.shape > 0 and data_df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_preprocess_data(data):
    '''
    test perform preprocess_data function
    '''
    try:
        # removing mocked columns
        data = data.drop(columns=['Churn'])

        preprocessed_data = cl.preprocess_data(data)
        assert 'Churn' in preprocessed_data.columns
        logging.info("Testing preprocess_data: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing preprocess_data: Churn column seems it was not created")
        raise err


def test_eda(data):
    '''
    test perform eda function
    '''
    plots_paths = [
        'churn_histogram.jpeg', 'customer_age.jpeg', 'corr_plot.jpeg',
        'marital_status_count.jpeg', 'total_trans_ct_kernel_density.jpeg',
        'scatter_open_to_buy_utilization.jpeg']

    # removing all images before testing
    files = glob.glob('./images/*')
    for file in files:
        os.remove(file)

    # performs eda/save images
    cl.perform_eda(data)
    try:
        for plot_path in plots_paths:
            assert os.path.isfile('./images/' + plot_path)

        logging.info("Testing perform_eda: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing perform_eda: some plots are missing")
        raise err


def test_encoder_helper(data, category_lst, response):
    '''
    test encoder helper
    '''
    try:
        # Call encoder helper
        data = cl.encoder_helper(data, category_lst, response)

        for cat in category_lst:
            assert f'{cat}_{response}' in data.columns
        logging.info("Testing encoder_helper: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing enconder_helper: missing encoded categorical columns")
        raise err


def test_perform_feature_engineering(data, response):
    '''
    test perform_feature_engineering
    '''

    try:
        data = cl.preprocess_data(data)

        X_train, X_test, y_train, y_test = cl.perform_feature_engineering(
            data, response)

        assert X_train.shape[0] > 0 and X_train.shape[1] > 1
        assert X_test.shape[0] > 0 and X_test.shape[1] > 1
        assert y_train.shape[0] > 0
        assert y_train.shape[0] > 0
        logging.info("Testing perform_feature_engineering: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing perform_feature_engineering: some of X_train, X_test, y_train, y_test data are empty")
        raise err


def test_train_models(train_test_data):
    '''
    test train_models
    '''

    X_train, X_test, y_train, y_test = train_test_data

    paths = ['./models/rfc_model.pkl', './models/logistic_model.pkl',
             './images/models_roc_curve_comparison.jpeg']

    try:
        cl.train_models(X_train, X_test, y_train, y_test)
        for path in paths:
            assert os.path.isfile(path)
        logging.info("Testing test_train_models: SUCCESS")
    except AssertionError as err:
        logging.error(
            'Testing train_models: models or images were not created')
        logging.error(err)
        raise err


def test_model_evaluation(model_evaluation_data):
    '''
    test model_evaluation
    '''
    y_train, y_test, y_train_preds_lr, y_train_preds_rf, \
        y_test_preds_lr, y_test_preds_rf = model_evaluation_data

    paths = ['./images/rf_classification_report.jpeg',
             './images/lr_classification_report.jpeg']

    try:
        cl.model_evaluation(y_train, y_test, y_train_preds_lr,
                            y_train_preds_rf, y_test_preds_lr,
                            y_test_preds_rf)
        for path in paths:
            assert os.path.isfile(path)

        logging.info("Testing model_evaluation: SUCCESS")
    except AssertionError as err:
        logging.error('Testing model_evaluation: images were not created')
        logging.error(err)
        raise err


def test_feature_importance_plot(rfc_model, train_test_data):
    '''
    test feature_importance_plot
    '''
    data_input, _, _, _ = train_test_data

    plots_paths = [
        'normal_feature_importance.jpeg', 'shap_feature_importance.jpeg']

    # removing all images before testing
    files = glob.glob('./images/*')
    for file in files:
        os.remove(file)

    try:
        cl.feature_importance_plot(rfc_model, data_input, output_pth='./images')

        for plot_path in plots_paths:
            assert os.path.isfile('./images/' + plot_path)
        logging.info("Testing feature_importance_plot: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing feature_importance_plot: some plots are missing")
        raise err


def test_predict(rfc_model, train_test_data):
    '''
    test predict function
    '''
    _, data_input, _, _ = train_test_data

    try:
        preds = cl.predict(
            model=rfc_model,
            X_data=data_input,
            output_pth='./data/predictions_test.csv')

        assert os.path.isfile('./data/predictions_test.csv')
        assert len(data_input) == len(preds)

        logging.info("Testing predict: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing predict: prediction file is missing or length predictions is different than input size")
        raise err
