import os
import sys
import pandas as pd
import numpy as np
from src.exception import CustomException
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV


def save_object(file_path,object):
    try:
        dir_path = os.path.dirname(file_path) # It will take the file Path

        os.makedirs(dir_path,exist_ok=True) # It will make the directory

        with open(file_path,"wb") as file_obj: #it open the file path in write byte mode
            dill.dump(object,file_obj)  # dill is another library help us to create Pickle file

    except Exception as e:
            raise CustomException(e,sys)

def evaluate_model(X_train,y_train,X_test,y_test,models,param):
    try:
        report = {}
        for i in range(len(list(models))):  # Going though Each model
            model = list(models.values())[i]
            param = param[list(param.keys())[i]]

            gs = GridSearchCV(model,param,cv=3)  # Get the best parameter for each model
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)   # Training model with best parameter

            #model.fit(X_train,y_train) # Training model

            y_train_pred = model.predict(X_train)  # Predicting y_train value for provided x_train
            y_test_pred = model.predict(X_test)    # Predicting y_test value for provided x_test

            train_model_score = r2_score(y_train,y_train_pred)  # Calculating r2 score by comparing actual y train and predicted y train value
            test_model_score = r2_score(y_test,y_test_pred)    # Calculating r2 score by comparing actual y test and predicted y test value

            report[list(models.keys())[i]] = test_model_score

            return report




    except Exception as e:
        raise CustomException(e,sys)
