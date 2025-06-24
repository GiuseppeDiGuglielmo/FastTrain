import os
import requests
import numpy as np
import time
from sklearn.datasets import fetch_openml

server = "127.0.0.1"
port = "8000"
base_url = f"http://{server}:{port}"

# Fetch data from OpenML
print("📥 Fetching data from OpenML...")
if not os.path.exists("data.npy") or not os.path.exists("labels.npy"):
    try:
        dataset = fetch_openml('hls4ml_lhc_jets_hlf', version=1, as_frame=False)
        X, y = dataset['data'], dataset['target']
        np.save("data.npy", X)
        np.save("labels.npy", y)
        print(f"✅ Data downloaded and saved. Shape: {X.shape}, Labels: {len(np.unique(y))}")
    except Exception as e:
        print(f"❌ Error fetching data: {e}")
        exit(1)
else:
    print("✅ Data files already exist locally")

# Upload data to training server
print("📤 Uploading data to training server...")
try:
    with open("data.npy", "rb") as data_file, open("labels.npy", "rb") as labels_file:
        response = requests.post(f"{base_url}/upload/", files={"data": data_file, "labels": labels_file})
        response.raise_for_status()
        print(f"✅ Upload successful: {response.json()}")
except requests.RequestException as e:
    print(f"❌ Error uploading data: {e}")
    exit(1)
except Exception as e:
    print(f"❌ Error reading local files: {e}")
    exit(1)

# Start training
print("🚀 Starting training...")
try:
    response = requests.post(f"{base_url}/start_training/")
    response.raise_for_status()
    result = response.json()
    print(f"✅ Training started: {result}")
    
    if result.get("status") == "already running":
        print("⚠️ Training is already in progress")
except requests.RequestException as e:
    print(f"❌ Error starting training: {e}")
    exit(1)

# Wait and check status with polling
print("⏳ Monitoring training status...")
max_wait_time = 300  # 5 minutes timeout
check_interval = 10  # Check every 10 seconds
elapsed_time = 0

while elapsed_time < max_wait_time:
    try:
        response = requests.get(f"{base_url}/status/")
        response.raise_for_status()
        status = response.json()
        
        if status.get("completed"):
            print("✅ Training completed successfully!")
            break
        elif status.get("running"):
            print(f"🔄 Training in progress... (elapsed: {elapsed_time}s)")
        else:
            print(f"📊 Status: {status}")
            
        time.sleep(check_interval)
        elapsed_time += check_interval
        
    except requests.RequestException as e:
        print(f"❌ Error checking status: {e}")
        break
else:
    print("⏰ Training timeout reached")

# Download model
print("📥 Downloading trained model...")
try:
    response = requests.get(f"{base_url}/get_model/")
    response.raise_for_status()
    
    if response.headers.get('content-type') == 'application/json':
        # Server returned JSON error instead of model file
        error_msg = response.json()
        print(f"❌ Model download failed: {error_msg}")
    else:
        with open("model_downloaded.keras", "wb") as f:
            f.write(response.content)
        print("✅ Model downloaded successfully as 'model_downloaded.keras'")
        print(f"📁 Model file size: {len(response.content)} bytes")
        
except requests.RequestException as e:
    print(f"❌ Error downloading model: {e}")
except Exception as e:
    print(f"❌ Error saving model file: {e}")