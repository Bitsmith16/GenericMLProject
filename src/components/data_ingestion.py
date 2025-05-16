import os
import sys
from src.exception import CustomException
from src.logger import logging

import pandas as pd 
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

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
    obj.initiate_data_initiation()



