import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost.sklearn import XGBClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
import itertools

from utils import TrainedModel

def train_LogisticRegression(X_train, y_train):
    model = LogisticRegression()
    model.fit(X_train, y_train)

    trained_model = TrainedModel(model_name='LR', trained_model=model)
    return trained_model

def train_KNN(X_train, y_train):
    model = KNeighborsClassifier()
    model.fit(X_train, y_train)

    trained_model = TrainedModel(model_name='KNN', trained_model=model)
    return trained_model

def train_DecisionTree(X_train, y_train):
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)

    trained_model = TrainedModel(model_name='DT', trained_model=model)
    return trained_model

def train_SVM(X_train, y_train):
    model = SVC(probability=True)
    model.fit(X_train, y_train)

    trained_model = TrainedModel(model_name='SVM', trained_model=model)
    return trained_model

def train_RandomForest(X_train, y_train):
    model = RandomForestClassifier(random_state=42, n_estimators=400)
    model.fit(X_train, y_train)

    trained_model = TrainedModel(model_name='RF', trained_model=model)
    return trained_model

def train_XGBoost(X_train, y_train):
    model = XGBClassifier(eval_metric='logloss')
    model.fit(X_train, y_train)

    trained_model = TrainedModel(model_name='XGB', trained_model=model)
    return trained_model
