import logging 
import os
from datetime import datetime


LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log" # Name formal of log file

Logs_path = os.path.join(os.getcwd(),"logs",LOG_FILE) # Log path

os.makedirs(Logs_path,exist_ok=True)

LOG_FILE_PATH = os.path.join(Logs_path,LOG_FILE) # Log file path where file will be stored

logging.basicConfig(
filename = LOG_FILE_PATH,
format ="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
level =logging.INFO,
)
