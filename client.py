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

# Fetch data from OpenML
logger.info("Fetching data from OpenML...")
if not os.path.exists("data.npy") or not os.path.exists("labels.npy"):
    try:
        dataset = fetch_openml('hls4ml_lhc_jets_hlf', version=1, as_frame=False)
        X, y = dataset['data'], dataset['target']
        np.save("data.npy", X)
        np.save("labels.npy", y)
        logger.info(f"Data downloaded and saved. Shape: {X.shape}, Labels: {len(np.unique(y))}")
    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        exit(1)
else:
    logger.info("Data files already exist locally")

# Upload data to training server
logger.info("Uploading data to training server...")
try:
    with open("data.npy", "rb") as data_file, open("labels.npy", "rb") as labels_file:
        response = requests.post(f"{base_url}/upload/", files={"data": data_file, "labels": labels_file})
        response.raise_for_status()
        logger.info(f"Upload successful: {response.json()}")
except requests.RequestException as e:
    logger.error(f"Error uploading data: {e}")
    exit(1)
except Exception as e:
    logger.error(f"Error reading local files: {e}")
    exit(1)

# Start training
logger.info("Starting training...")
try:
    response = requests.post(f"{base_url}/start_training/")
    response.raise_for_status()
    result = response.json()
    logger.info(f"Training started: {result}")
    
    if result.get("status") == "already running":
        logger.info("Training is already in progress")
except requests.RequestException as e:
    logger.error(f"Error starting training: {e}")
    exit(1)

# Wait and check status with polling
logger.info("Monitoring training status...")
max_wait_time = 300  # 5 minutes timeout
check_interval = 10  # Check every 10 seconds
elapsed_time = 0

while elapsed_time < max_wait_time:
    try:
        response = requests.get(f"{base_url}/status/")
        response.raise_for_status()
        status = response.json()
        
        if status.get("completed"):
            logger.info("Training completed successfully!")
            break
        elif status.get("running"):
            logger.info(f"Training in progress... (elapsed: {elapsed_time}s)")
        else:
            logger.info(f"Status: {status}")
            
        time.sleep(check_interval)
        elapsed_time += check_interval
        
    except requests.RequestException as e:
        logger.error(f"Error checking status: {e}")
        break
else:
    logger.info("Training timeout reached")

# Download model
logger.info("Downloading trained model...")
try:
    response = requests.get(f"{base_url}/get_model/")
    response.raise_for_status()
    
    if response.headers.get('content-type') == 'application/json':
        # Server returned JSON error instead of model file
        error_msg = response.json()
        logger.error(f"Model download failed: {error_msg}")
    else:
        with open("model_downloaded.keras", "wb") as f:
            f.write(response.content)
        logger.info("Model downloaded successfully as 'model_downloaded.keras'")
        logger.info(f"Model file size: {len(response.content)} bytes")
        
except requests.RequestException as e:
    logger.error(f"Error downloading model: {e}")
except Exception as e:
    logger.error(f"Error saving model file: {e}")
