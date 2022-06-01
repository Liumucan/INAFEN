import argparse
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from utils import EvaluationRecorder, ResultsSummary, preprocessing_data, makedir, record_results_summary, ParameterDict
from train_baseline_models import train_LogisticRegression, train_RandomForest, train_XGBoost, train_SVM, train_KNN, train_DecisionTree

from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


import os
import pickle
import json
import time

import warnings
warnings.filterwarnings('ignore')


base_model_dict = {
    'XGB': XGBClassifier,
    'LR': LogisticRegression,
    'RF': RandomForestClassifier,
    'DT': DecisionTreeClassifier,
    'KNN': KNeighborsClassifier,
    'SVM': SVC}

baseline_parameter_dict = {
    'XGB': {'n_estimators': [100, 200, 300],
            'max_depth': [2, 3],
            'gamma': [0, 0.5, 1]},
    'RF': {'n_estimators': [100, 200, 300],
           'max_depth': [2, 3],
           'min_samples_split': [2, 3]},
    'LR': {'C': [0.01, 0.1, 1, 10],
           'penalty': ['l2']},
    'DT': {'max_depth': [2, 3, 4, 5],
           'criterion': ['entropy', 'gini'],
           'min_samples_leaf': [5, 10, 20, 30]},
    'KNN': {'n_neighbors': [2, 4, 6, 8, 10]},
    'SVM': {'C': [0.1, 1, 10],
            'gamma': [1, 0.1, 0.01],
            'probability': [True]}}

if __name__ == '__main__':
    desc = "Algorithm Command Line Tool and Library"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--data_path', type=str, default='./Data')
    parser.add_argument('--save_path', type=str, default='./Results/')
    parser.add_argument('--dataset_name', type=str, default='dataset_31_credit-g.csv')
    parser.add_argument('--baseline_model', type=str, default='LR')

    args = parser.parse_args()
    print('Dataset: {}, Baseline models: {}'.format(args.dataset_name, args.baseline_model))

    # Data reading & preprocessing
    data_df = pd.read_csv(os.path.join(args.data_path, 'Raw_data', args.dataset_name))

    data_df, num_categorical_feas, num_numerical_feas = preprocessing_data(data_df, args.dataset_name)
    data_df_columns = data_df.columns[:-1]

    X = data_df.iloc[:, :-1].values
    y = data_df.iloc[:, -1].values

    save_path = os.path.join(args.save_path, args.dataset_name.split('.')[0], args.baseline_model)
    makedir(save_path=save_path)

    baseline_parameter = ParameterDict(baseline_parameter_dict)
    parameter_list = baseline_parameter.get_parameter_list(args.baseline_model)

    for parameter_dict in parameter_list:
        print(parameter_dict)
        valid_evaluation_recorder = EvaluationRecorder()
        test_evaluation_recorder = EvaluationRecorder()
        valid_evaluation_recorder.add_model(args.baseline_model)
        test_evaluation_recorder.add_model(args.baseline_model)
        skf_train_test = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for i, (train_index, test_index) in enumerate(skf_train_test.split(X, y)):
            print('Using data of fold {}...'.format(i + 1))
            X_train, X_test = X[train_index, :], X[test_index, :]
            y_train, y_test = y[train_index], y[test_index]

            skf_train_valid = StratifiedKFold(n_splits=4, shuffle=False)
            train_train_index, train_valid_index = list(skf_train_valid.split(X_train, y_train))[0]
            X_train_train, X_train_valid = X_train[train_train_index, :], X_train[train_valid_index, :]
            y_train_train, y_train_valid = y_train[train_train_index], y_train[train_valid_index]

            time_start = time.time()
            model = base_model_dict[args.baseline_model](**parameter_dict)
            model.fit(X_train_train, y_train_train)
            # model.fit(X_train, y_train)
            time_end = time.time()

            valid_evaluation_recorder.add_model_evaluation(args.baseline_model, model, (X_train_valid, y_train_valid))
            test_evaluation_recorder.add_model_evaluation(args.baseline_model, model, (X_test, y_test))
            valid_evaluation_recorder.evaluation_dict[args.baseline_model]['run_time'].append(time_end-time_start)
            test_evaluation_recorder.evaluation_dict[args.baseline_model]['run_time'].append(time_end - time_start)

        file_path = os.path.join(save_path, 'AUC[{:.4f}]-Param[{}].txt'.format(np.array(valid_evaluation_recorder.evaluation_dict[args.baseline_model]['auroc']).mean(),
                                                                                parameter_dict).replace(':', '-'))
        with open(file_path, 'w') as f:
            f.write('Model Name: {}\n'.format(args.baseline_model))
            f.write('Model Params: {}\n'.format(parameter_dict))
            f.write('Res. of validation set:\n')
            for metric in valid_evaluation_recorder.evaluation_dict[args.baseline_model].keys():
                f.write('{}: {:.4f}+-{:.4f}\n'.format(metric,
                                                     np.array(valid_evaluation_recorder.evaluation_dict[args.baseline_model][metric]).mean(),
                                                     np.array(valid_evaluation_recorder.evaluation_dict[args.baseline_model][metric]).std()))

            f.write('\n')
            f.write('Res. of test set:\n')
            for metric in test_evaluation_recorder.evaluation_dict[args.baseline_model].keys():
                f.write('{}: {:.4f}+-{:.4f}\n'.format(metric,
                                                     np.array(test_evaluation_recorder.evaluation_dict[args.baseline_model][metric]).mean(),
                                                     np.array(test_evaluation_recorder.evaluation_dict[args.baseline_model][metric]).std()))














