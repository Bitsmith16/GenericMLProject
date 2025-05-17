## Will train different model and see their performance
import os
import sys
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_model


from sklearn.linear_model import LinearRegression
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)

from sklearn.metrics import r2_score

@dataclass
class ModelTrainerConfig():
    trained_model_file_path = os.path.join("artificats","model.pkl")

class ModelTrainer():
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_trainer(self,train_array,test_array,preprocessor_path):

        try:
            logging.info("Spliting Training and Test Input data")
            X_train,y_train,X_test,y_test =(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            model =  {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting":GradientBoostingRegressor(),
                "Linear Regression":LinearRegression(),
                "K_Neighbors Classifier": KNeighborsRegressor(),
                "XGBClassifier": XGBRegressor(),
                "CatBoosting Classifier":CatBoostRegressor(verbose =False),
                "AdaBoost Classifier":AdaBoostRegressor()
            }
            model_report:dict =evaluate_model(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=model) # Defining function in Util

            best_model_score = max(sorted(model_report.values())) # To get best model score from model report dictionary

            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)] #To get best model name from model report dictionary

            best_model = model[best_model_name]

            if best_model_score <0.6: # No model is having score of 60% or more
                raise CustomException("No best Model Found")  # And we need to do hypertuning os some other technique so the model performace will increase
            logging.info(f"Best model found on both training and test dataset")

            save_object(      # Savinf model path
                file_path = self.model_trainer_config.trained_model_file_path,
                object = best_model # Creating pickel file of best model
            )

            predicted = best_model.predict(X_test) # Checking predicted value of y test with respect to best model , this same code

            r2_square = r2_score(y_test,predicted) # calculating r2 score of y_test with respct to predicted y value

            return r2_square ,best_model


        except Exception as e:
            raise CustomException(e,sys)

