## Do the Feature Engineering , Data Cleaning (Transforming Data)

import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer  ## ColumnTransformer is used to create pipeline
from sklearn.impute import SimpleImputer      ## For handling Missing value
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object


@dataclass
class Data_transformation_config:
    preprocessor_obj_file_path=os.path.join("artifacts","preprocessor.pkl")  ##To save created model as Python pickle file , this will be the path where we will stored the model


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = Data_transformation_config()
    
    def get_data_transformer(self):
        logging.info("Entered the Data Transformation Method or component")
        try:
            numerical_columns = ["writing_score","reading_score"]

            categorical_columns=["gender","race_ethnicity","parental_level_of_education","lunch","test_preparation_course"]

            numerical_pipeline = Pipeline(                    #There are 2 steps - Transforming Numerical column with Imputer for missing value & Standard scalar
                steps=[
                    ("imputer",SimpleImputer(strategy='median')),
                    ("scalar",StandardScaler(with_mean=False))
                ]
            )

            logging.info(f"Numerical Feature:{numerical_columns}")
            logging.info(f"Categorical Feature:{categorical_columns}")

            Categorical_pipeline = Pipeline(               #There are 3 steps - Transforming categorical column with Imputer for missing value,One hot Encoding & Standard Scalar
                steps=[
                    ("imputer",SimpleImputer(strategy='most_frequent')),
                    ("one_hot_encoder",OneHotEncoder(handle_unknown='ignore')),
                    ("scalar",StandardScaler(with_mean=False))
                ]
            )

            preprocessor = ColumnTransformer(   # Combining Numerical Pipeline and categorical pipeline and providing feature
                [("Numerical Pipeline",numerical_pipeline,numerical_columns),
                 ("Categorical Pipeline",Categorical_pipeline,categorical_columns)]
            )

            logging.info("Numerical & Categorical features are Transformed")

            return preprocessor


        except Exception as e:
            raise CustomException(e,sys)

    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df= pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Reading train and test data is completed")

            preprocessing_obj = self.get_data_transformer() #It return the preprocessor to preprocessing obj so that will be able to apply transformation on Train and test data later

            logging.info("Creating dependent(target/output) variable and Inpendent variable(input feature)")

            target_column_name="math_score"
            numerical_columns = ["writing_score", "reading_score"]

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1) #Input Training feature
            target_feature_train_df=train_df[target_column_name]                      #Output Training feature

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1) #Input Test feature
            target_feature_test_df=test_df[target_column_name]                      #Output Test feature

            logging.info(f"Applying preprocessing object on training dataframe and testing dataframe.")

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)


            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]  #placing the input_feature_train_arr and the target_feature_train_df side-by-side as columns(concatenating both)
            # train_arr becomes a single NumPy array where: All input features are on the left columns The target feature(s) is/are appended as the last column(s)
            # Similarly below code for test data in array form
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]    #placing the input_feature_test_arr and the target_feature_test_df side-by-side as columns(concatenating both)

            logging.info(f"Saved preprocessing object.")

            save_object( # Funtion define in Utill , calling that function in order to save Pickle file passing below 2 parameter

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                object=preprocessing_obj  # Saving Transformed data as pickel file under Artifcat folder with name preprocessor.pkl

            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

            
        except Exception as e:
            raise CustomException(e,sys)




