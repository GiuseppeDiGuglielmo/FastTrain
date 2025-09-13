import os
import requests
import numpy as np
import time
import logging
from sklearn.datasets import fetch_openml

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

#server = "127.0.0.1"
server = "131.225.220.30"
port = "8000"
base_url = f"http://{server}:{port}"

logger.info("Checking for local data files...")
if not os.path.exists("data.npy") or not os.path.exists("labels.npy"):
    try:
        logger.info("Downloading dataset from OpenML...")
        dataset = fetch_openml('hls4ml_lhc_jets_hlf', version=1, as_frame=False)
        X, y = dataset['data'], dataset['target']
        np.save("data.npy", X)
        np.save("labels.npy", y)
        logger.info(f"Dataset downloaded and saved locally. Data shape: {X.shape}, Number of classes: {len(np.unique(y))}")
    except Exception as e:
        logger.error(f"Failed to download dataset: {e}")
        exit(1)
else:
    logger.info("Local data files found. Skipping download.")

logger.info("Uploading data files to the training server...")
try:
    with open("data.npy", "rb") as data_file, open("labels.npy", "rb") as labels_file:
        response = requests.post(f"{base_url}/upload/", files={"data": data_file, "labels": labels_file})
        response.raise_for_status()
        logger.info(f"Data files uploaded successfully. Server response: {response.json()}")
except requests.RequestException as e:
    logger.error(f"Failed to upload data files: {e}")
    exit(1)
except Exception as e:
    logger.error(f"Failed to read local data files: {e}")
    exit(1)

logger.info("Requesting model training to start...")
try:
    response = requests.post(f"{base_url}/start_training/")
    response.raise_for_status()
    result = response.json()
    logger.info(f"Model training request sent. Server response: {result}")
    
    if result.get("status") == "already running":
        logger.info("Model training is already in progress.")
except requests.RequestException as e:
    logger.error(f"Failed to start model training: {e}")
    exit(1)

logger.info("Polling training status from server...")
max_wait_time = 300  # 5 minutes timeout
check_interval = 10  # Check every 10 seconds
elapsed_time = 0

while elapsed_time < max_wait_time:
    try:
        response = requests.get(f"{base_url}/status/")
        response.raise_for_status()
        status = response.json()
        
        if status.get("completed"):
            logger.info("Model training completed successfully.")
            break
        elif status.get("running"):
            logger.info(f"Model training in progress. Elapsed time: {elapsed_time} seconds.")
        else:
            logger.info(f"Training status: {status}")
            
        time.sleep(check_interval)
        elapsed_time += check_interval
        
    except requests.RequestException as e:
        logger.error(f"Failed to retrieve training status: {e}")
        break
else:
    logger.info("Training status polling timed out after 5 minutes.")

logger.info("Requesting trained model download from server...")
try:
    response = requests.get(f"{base_url}/get_model/")
    response.raise_for_status()
    
    if response.headers.get('content-type') == 'application/json':
        error_msg = response.json()
        logger.error(f"Failed to download trained model. Server response: {error_msg}")
    else:
        with open("model_downloaded.keras", "wb") as f:
            f.write(response.content)
        logger.info("Trained model downloaded successfully as 'model_downloaded.keras'.")
        logger.info(f"Downloaded model file size: {len(response.content)} bytes.")
        
except requests.RequestException as e:
    logger.error(f"Failed to download trained model: {e}")
except Exception as e:
    logger.error(f"Failed to save trained model file: {e}")
