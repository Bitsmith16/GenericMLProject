# Will take the Data
import os
import sys
from src.exception import CustomException
from src.logger import logging

import pandas as pd 
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import Data_transformation_config

from src.components.model_trainer import ModelTrainer
from src.components.model_trainer import ModelTrainerConfig


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifacts","train.csv")
    test_data_path: str = os.path.join("artifacts","test.csv")
    raw_data_path: str = os.path.join("artifacts","rawdata.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_data_initiation(self):
        logging.info("Entered the Data Ingestion Method or component")
        try:                                           # Writing all code in Try & Except block to catch error
            df=pd.read_csv("notebook\stud.csv")        # Reading data from local File
            logging.info("Entered the Data Ingestion Method or component")  # Read Data
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True) # It will create Artifact Folder to store train , test and raw data

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True) # Storing raw data 

            logging.info("Stored Raw data")
            logging.info("Train test split initiated")

            train_set,test_set = train_test_split(df,test_size=0.2,random_state=42) # test data is 20% and train data is 80 %

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True) # Adding train file in Artifact folder

            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True) # Adding test file in Artifact folder

            logging.info("Ingestion of data is completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e,sys)

if __name__=="__main__":
    obj = DataIngestion()
    train_data,test_data= obj.initiate_data_initiation() ## Data Ingestion return Train & Test Data

    obj2 = DataTransformation() # creating object of DataTransformation class so call in next line initiate_data_transformation function
    train_array,test_array,preprocessor_path = obj2.initiate_data_transformation(train_data,test_data) # initiate_data_transformation from dataTransformation class return 3 values which we are storing on left side variable train_array,test_array,preprocessor_path

    obj3 = ModelTrainer()
    r2_score,best_model=obj3.initiate_model_trainer(train_array,test_array,preprocessor_path) # though we are not using preprocessor_path parameter in function 
    print(best_model , r2_score)  # printing r2 score that the function has returned in above line



