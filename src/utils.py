import os
import sys
import pandas as pd
import numpy as np
from src.exception import CustomException
import dill


def save_object(file_path,object):
    try:
        dir_path = os.path.dirname(file_path) # It will take the file Path

        os.makedirs(dir_path,exist_ok=True) # It will make the directory

        with open(file_path,"wb") as file_obj: #it open the file path in write byte mode
            dill.dump(object,file_obj)  # dill is another library help us to create Pickle file

    except Exception as e:
            raise CustomException(e,sys)
