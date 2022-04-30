# library doc string
"""
Module containing all the helper functions to run the churn prediction pipeline

Author: Leonardo Meireles
Data: 15/04/2022
"""


# import libraries
import os
import logging
import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

sns.set()

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

os.environ['QT_QPA_PLATFORM'] = 'offscreen'


# Global/Constant Variables
CAT_COLUMNS = [
    'Gender',
    'Education_Level',
    'Marital_Status',
    'Income_Category',
    'Card_Category'
]

QUANT_COLUMNS = [
    'Customer_Age',
    'Dependent_count',
    'Months_on_book',
    'Total_Relationship_Count',
    'Months_Inactive_12_mon',
    'Contacts_Count_12_mon',
    'Credit_Limit',
    'Total_Revolving_Bal',
    'Avg_Open_To_Buy',
    'Total_Amt_Chng_Q4_Q1',
    'Total_Trans_Amt',
    'Total_Trans_Ct',
    'Total_Ct_Chng_Q4_Q1',
    'Avg_Utilization_Ratio'
]

KEEP_COLS = ['Customer_Age', 'Dependent_count', 'Months_on_book',
             'Total_Relationship_Count', 'Months_Inactive_12_mon',
             'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
             'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
             'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
             'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn',
             'Income_Category_Churn', 'Card_Category_Churn']


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
        pth: a path to the csv
    output:
        data_df: pandas dataframe
    '''
    if not os.path.isfile(pth):
        raise FileNotFoundError

    data_df = pd.read_csv(pth)

    return data_df


def preprocess_data(data_df):
    '''
    returns dataframe with initial preprocessing needed (not feature engineering)

    input:
        data_df: pd Dataframe
    output:
        data_df: pandas dataframe
    '''
    data_df['Churn'] = data_df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    return data_df


def perform_eda(data_df):
    '''
    perform eda on data_df and save figures to images folder
    input:
        data_df: pandas dataframe

    output:
        None
    '''
    logging.info('DataFrame Shape: %s\n', data_df.shape)
    logging.info('Feature Null Count: \n%s', data_df.isnull().sum())
    logging.info('Descriptive metrics:\n%s', data_df.describe())

    # Viz EDA

    # Churn Histogram
    plt.figure(figsize=(20, 10))
    data_df['Churn'].hist()
    plt.title('Churn Histogram')
    plt.savefig('./images/churn_histogram.jpeg')
    plt.close()

    # Customer Age hist
    plt.figure(figsize=(20, 10))
    data_df['Customer_Age'].hist()
    plt.title('Customer Age Histogram')
    plt.savefig('./images/customer_age.jpeg')
    plt.close()

    # Marital Status Count Plot
    plt.figure(figsize=(20, 10))
    sns.countplot(data=data_df, x='Marital_Status')
    plt.title('Marital Status Count Plot')
    plt.savefig('./images/marital_status_count.jpeg')
    plt.close()

    plt.figure(figsize=(20, 10))
    # Show distributions of 'Total_Trans_Ct' and add a smooth curve obtained
    # using a kernel density estimate
    sns.histplot(data_df['Total_Trans_Ct'], stat='density', kde=True)
    plt.title('Total Transactions Ct')
    plt.savefig('./images/total_trans_ct_kernel_density.jpeg')
    plt.close()

    plt.figure(figsize=(20, 10))
    sns.scatterplot(data=data_df, x='Avg_Open_To_Buy', y='Avg_Utilization_Ratio')
    plt.title('Avg Open To Buy vs Avg Utilization Ratio')
    plt.savefig('./images/scatter_open_to_buy_utilization.jpeg')
    plt.close()

    plt.figure(figsize=(20, 10))
    sns.heatmap(data_df.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.savefig('./images/corr_plot.jpeg')
    plt.close()


def encoder_helper(data_df, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
        data_df: pandas dataframe
        category_lst: list of columns that contain categorical features
        response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
        data_df: pandas dataframe with new columns for
    '''
    for cat in category_lst:
        data_df = data_df.merge(data_df.groupby(cat).mean()[response].to_frame(
            f'{cat}_{response}').reset_index(), on=cat)

    return data_df


def perform_feature_engineering(data_df, response):
    '''
    input:
          data_df: pandas dataframe
          response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
          X_train: X training data
          X_test: X testing data
          y_train: y training data
          y_test: y testing data
    '''

    y = data_df[response]

    # train test split
    X_train, X_test, y_train, y_test = train_test_split(
        data_df.drop(columns=[response]), y, test_size=0.3, random_state=42)

    # Prevents data leakage
    X_train = encoder_helper(X_train, CAT_COLUMNS, response)
    X_test = encoder_helper(X_test, CAT_COLUMNS, response)
    X_train = X_train[KEEP_COLS].copy()
    X_test = X_test[KEEP_COLS].copy()

    # train test split
    return X_train, X_test, y_train, y_test


def model_evaluation(y_train,
                     y_test,
                     y_train_preds_lr,
                     y_train_preds_rf,
                     y_test_preds_lr,
                     y_test_preds_rf):
    '''
    produces model evaluation containing classification report for
    training and testing results and stores report
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
    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Random Forest Train'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(
        0.01, 0.05, str(classification_report(y_test, y_test_preds_rf)),
        {'fontsize': 10},
        fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Random Forest Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(
        0.01, 0.7, str(classification_report(y_train, y_train_preds_rf)),
        {'fontsize': 10},
        fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig('./images/rf_classification_report.jpeg')
    plt.close()

    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Logistic Regression Train'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(
        0.01, 0.05, str(classification_report(y_train, y_train_preds_lr)),
        {'fontsize': 10},
        fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Logistic Regression Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(
        0.01, 0.7, str(classification_report(y_test, y_test_preds_lr)),
        {'fontsize': 10},
        fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig('./images/lr_classification_report.jpeg')
    plt.close()


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
        model: model object containing feature_importances_
        X_data: pandas dataframe of X values
        output_pth: path to store the figure

    output:
         None
    '''

    explainer = shap.TreeExplainer(
        model)
    shap_values = explainer.shap_values(X_data)
    shap.summary_plot(shap_values, X_data, plot_type="bar")
    plt.savefig(
        os.path.join(
            output_pth, 'shap_feature_importance.jpeg'))
    plt.close()

    # Calculate feature importances
    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90)
    plt.savefig(
        os.path.join(
            output_pth, 'normal_feature_importance.jpeg'))
    plt.close()


def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
          X_train: X training data
          X_test: X testing data
          y_train: y training data
          y_test: y testing data
    output:
          y_train: pd.Series
          y_test: pd.Series
          y_train_preds_lr: array/np.array
          y_train_preds_rf: array/np.array
          y_test_preds_lr: array/np.array
          y_test_preds_rf: array/np.array
    '''
    # This cell may take up to 15-20 minutes to run
    # grid search
    rfc = RandomForestClassifier(random_state=42)
    # Use a different solver if the default 'lbfgs' fails to converge
    # Reference:
    # https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(
        estimator=rfc, param_grid=param_grid, cv=5, n_jobs=-1)
    cv_rfc.fit(X_train, y_train)

    lrc.fit(X_train, y_train)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    lrc_plot = plot_roc_curve(lrc, X_test, y_test)

    # plots
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    _ = plot_roc_curve(cv_rfc.best_estimator_,
                              X_test, y_test, ax=ax, alpha=0.8)
    lrc_plot.plot(ax=ax, alpha=0.8)
    plt.savefig('./images/models_roc_curve_comparison.jpeg')
    plt.close()

    # Saving models
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')

    return y_train, y_test, y_train_preds_lr, y_train_preds_rf, \
        y_test_preds_lr, y_test_preds_rf


def predict(model, X_data, output_pth):
    '''
    Churn predictions given the data in X_data
    input:
        model: model object containing feature_importances_
        X_data: pandas dataframe of X values
        output_pth: path to store predictions in csv

    output:
         pred_df: Pandas DataFrame containing the predictions
    '''

    predictions = model.predict(X_data)

    pred_df = pd.DataFrame(predictions, columns=['prediction'])

    pred_df.to_csv(output_pth, index=False)

    return pred_df


if __name__ == 'main':
    data_df = import_data('./data/bank_data.csv')
    data_df = preprocess_data(data_df)
    perform_eda(data_df)
    X_train, X_test, y_train, y_test = perform_feature_engineering(
        data_df, response='Churn')

    _, _, y_train_preds_lr, y_train_preds_rf, \
        y_test_preds_lr, y_test_preds_rf = train_models(X_train, X_test,
                                                        y_train, y_test)

    rfc_model = joblib.load('./models/rfc_model.pkl')
    logistic_model = joblib.load('./models/logistic_model.pkl')

    model_evaluation(y_train, y_test, y_train_preds_lr, y_train_preds_rf,
                     y_test_preds_lr, y_test_preds_rf)

    feature_importance_plot(model=rfc_model, X_data=X_test,
                            output_pth='./images/')
