from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
import uvicorn
import os
import threading
import time
import logging
import traceback
from datetime import datetime

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0=all, 1=info, 2=warning, 3=error
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='You are saving your model as an HDF5 file*')

import tensorflow as tf
# Additional TensorFlow logging suppression
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(0)

import numpy as np

from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from tensorflow.keras.layers import Dense, Activation, BatchNormalization, Dropout, Input, Add
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1
from callbacks import all_callbacks

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('server.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

#server = "127.0.0.1"
server = "131.225.220.30"
port = 8000

app = FastAPI(title="FastTrain Server", description="FastAPI server for remote ML training")
DATA_DIR = "uploaded_data"
MODEL_PATH = "trained_model.keras"
os.makedirs(DATA_DIR, exist_ok=True)

training_status = {
    "running": False, 
    "completed": False, 
    "error": None,
    "progress": 0,
    "start_time": None,
    "end_time": None,
    "current_epoch": 0,
    "total_epochs": 10,
    "metrics": {}
}

def reset_training_status():
    """Reset training status to initial state"""
    training_status.update({
        "running": False,
        "completed": False,
        "error": None,
        "progress": 0,
        "start_time": None,
        "end_time": None,
        "current_epoch": 0,
        "total_epochs": 10,
        "metrics": {}
    })
    logger.info("Training status has been reset to initial state.")

def build_model(input_dim, num_classes):
    """Build the current simple model (baseline)."""
    model = Sequential()
    model.add(Dense(64,
                    input_shape=(input_dim,),
                    name='fc1',
                    kernel_initializer='lecun_uniform',
                    kernel_regularizer=l1(0.0001)))
    model.add(Activation(
                    activation='relu',
                    name='relu1'))
    model.add(Dense(32, name='fc2',
                    kernel_initializer='lecun_uniform',
                    kernel_regularizer=l1(0.0001)))
    model.add(Activation(
                    activation='relu',
                    name='relu2'))
    model.add(Dense(32,
                    name='fc3',
                    kernel_initializer='lecun_uniform',
                    kernel_regularizer=l1(0.0001)))
    model.add(Activation(
                    activation='relu',
                     name='relu3'))
    model.add(Dense(num_classes, name='output',
                    kernel_initializer='lecun_uniform',
                    kernel_regularizer=l1(0.0001)))
    model.add(Activation(
                    activation='softmax',
                    name='softmax'))
    return model

def build_complex_model(input_dim, num_classes):
    """Build a more complex model to stress the GH200 GPU."""
    inputs = Input(shape=(input_dim,))
    x = Dense(512, kernel_initializer='he_normal', kernel_regularizer=l1(0.0001))(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.3)(x)

    # Residual block 1
    shortcut = x
    x = Dense(512, kernel_initializer='he_normal', kernel_regularizer=l1(0.0001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.3)(x)
    x = Add()([x, shortcut])

    # Residual block 2
    shortcut = x
    x = Dense(256, kernel_initializer='he_normal', kernel_regularizer=l1(0.0001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.3)(x)
    # Project shortcut to match x's shape if needed
    shortcut_proj = Dense(256, kernel_initializer='he_normal', kernel_regularizer=l1(0.0001))(shortcut)
    x = Add()([x, shortcut_proj])

    # More dense layers
    x = Dense(128, kernel_initializer='he_normal', kernel_regularizer=l1(0.0001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.3)(x)

    x = Dense(64, kernel_initializer='he_normal', kernel_regularizer=l1(0.0001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.3)(x)

    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model

def build_wide_model(input_dim, num_classes):
    """Build a wide model with many units per layer."""
    model = Sequential()
    model.add(Dense(1024, input_shape=(input_dim,), activation='relu', kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(1024, activation='relu', kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    return model

def train_model():
    """Train the model with error handling and progress tracking."""
    try:
        training_status["running"] = True
        training_status["start_time"] = datetime.now().isoformat()
        training_status["error"] = None
        logger.info("Loading training data...")

        # Load and validate data
        data_path = os.path.join(DATA_DIR, "data.npy")
        labels_path = os.path.join(DATA_DIR, "labels.npy")

        if not os.path.exists(data_path) or not os.path.exists(labels_path):
            logger.error("Training data files not found.")
            raise FileNotFoundError("Training data files not found.")

        data = np.load(data_path, allow_pickle=True)
        labels = np.load(labels_path, allow_pickle=True)
        logger.info(f"Training data loaded successfully. Data shape: {data.shape}, Labels shape: {labels.shape}")

        logger.info("Preprocessing training data...")
        # Data preprocessing
        le = LabelEncoder()
        labels = le.fit_transform(labels)
        labels = to_categorical(labels, 5)
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            data, labels, test_size=0.2, random_state=42
        )

        scaler = StandardScaler()
        X_train_val = scaler.fit_transform(X_train_val)
        X_test = scaler.transform(X_test)
        logger.info("Training data preprocessing completed.")

        logger.info("Building and training the model...")
        # Choose model here:
        #model = build_model(input_dim=16, num_classes=5)
        model = build_complex_model(input_dim=16, num_classes=5)
        #model = build_wide_model(input_dim=16, num_classes=5)

        training_status["progress"] = 10

        adam = Adam(learning_rate=0.0001)
        model.compile(optimizer=adam, loss=['categorical_crossentropy'], metrics=['accuracy'])

        class ProgressCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                training_status["current_epoch"] = epoch + 1
                training_status["progress"] = int((epoch + 1) / training_status["total_epochs"] * 80) + 10
                training_status["metrics"] = logs or {}
                logger.info(f"Epoch {epoch + 1}/{training_status['total_epochs']} completed. Metrics: {logs}")

        callbacks = all_callbacks(
            stop_patience=1000,
            lr_factor=0.5,
            lr_patience=10,
            lr_epsilon=0.000001,
            lr_cooldown=2,
            lr_minimum=0.0000001,
            outputDir='model_1',
        )
        callbacks.callbacks.append(ProgressCallback())

        history = model.fit(
            X_train_val,
            y_train_val,
            batch_size=1024,
            epochs=training_status["total_epochs"],
            validation_split=0.25,
            shuffle=True,
            callbacks=callbacks.callbacks,
            verbose=0
        )

        # Evaluate model
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        training_status["metrics"]["test_loss"] = test_loss
        training_status["metrics"]["test_accuracy"] = test_accuracy
        logger.info(f"Model evaluation completed. Test accuracy: {test_accuracy:.4f}")

        # Save model
        model.save(MODEL_PATH)
        training_status["progress"] = 100
        training_status["running"] = False
        training_status["completed"] = True
        training_status["end_time"] = datetime.now().isoformat()
        logger.info("Model training completed and model saved.")

    except Exception as e:
        error_msg = f"Model training failed: {str(e)}"
        training_status["running"] = False
        training_status["completed"] = False
        training_status["error"] = error_msg
        training_status["end_time"] = datetime.now().isoformat()
        logger.error(f"{error_msg}\n{traceback.format_exc()}")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "ML Training Server",
        "version": "1.0",
        "endpoints": {
            "/upload/": "POST - Upload training data",
            "/start_training/": "POST - Start model training",
            "/status/": "GET - Get training status",
            "/get_model/": "GET - Download trained model",
            "/reset/": "POST - Reset training status"
        }
    }

@app.post("/upload/")
async def upload_data(data: UploadFile = File(...), labels: UploadFile = File(...)):
    """Upload training data files with validation."""
    try:
        logger.info(f"Uploading files: {data.filename}, {labels.filename}")

        # Validate file extensions
        if not data.filename.endswith('.npy') or not labels.filename.endswith('.npy'):
            logger.error("Uploaded files must be in .npy format.")
            raise HTTPException(status_code=400, detail="Files must be .npy format.")

        # Save data file
        data_content = await data.read()
        with open(os.path.join(DATA_DIR, "data.npy"), "wb") as f:
            f.write(data_content)

        # Save labels file
        labels_content = await labels.read()
        with open(os.path.join(DATA_DIR, "labels.npy"), "wb") as f:
            f.write(labels_content)

        # Validate uploaded files
        try:
            data_array = np.load(os.path.join(DATA_DIR, "data.npy"), allow_pickle=True)
            labels_array = np.load(os.path.join(DATA_DIR, "labels.npy"), allow_pickle=True)

            logger.info(f"Files uploaded and validated successfully. Data shape: {data_array.shape}, Labels shape: {labels_array.shape}")

            return {
                "status": "uploaded",
                "data_shape": data_array.shape,
                "labels_shape": labels_array.shape,
                "data_size_mb": len(data_content) / (1024 * 1024),
                "labels_size_mb": len(labels_content) / (1024 * 1024)
            }
        except Exception as e:
            logger.error(f"Uploaded files are not valid numpy arrays: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Invalid numpy files: {str(e)}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Data upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/start_training/")
async def start_training():
    """Start model training with error handling."""
    try:
        if training_status["running"]:
            logger.info("Training is already in progress.")
            return {"status": "already running", "current_epoch": training_status["current_epoch"]}

        # Check if data files exist
        if not os.path.exists(os.path.join(DATA_DIR, "data.npy")) or not os.path.exists(os.path.join(DATA_DIR, "labels.npy")):
            logger.error("No training data found. Please upload data first.")
            raise HTTPException(status_code=400, detail="No training data uploaded. Please upload data first.")

        # Reset status and start training
        reset_training_status()
        training_thread = threading.Thread(target=train_model, daemon=True)
        training_thread.start()
        logger.info("Model training has started.")

        return {
            "status": "training started",
            "start_time": training_status["start_time"],
            "total_epochs": training_status["total_epochs"]
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start model training: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to start training: {str(e)}")

@app.get("/status/")
async def get_status():
    """Get detailed training status."""
    status_copy = training_status.copy()

    # Calculate training duration if available
    if status_copy["start_time"]:
        start_time = datetime.fromisoformat(status_copy["start_time"])
        if status_copy["end_time"]:
            end_time = datetime.fromisoformat(status_copy["end_time"])
            duration = (end_time - start_time).total_seconds()
        else:
            duration = (datetime.now() - start_time).total_seconds()
        status_copy["duration_seconds"] = round(duration, 2)

    #logger.info("Training status requested.")
    return status_copy

@app.post("/reset/")
async def reset_status():
    """Reset training status."""
    reset_training_status()
    logger.info("Training status reset via API endpoint.")
    return {"status": "reset", "message": "Training status has been reset."}

@app.get("/get_model/")
async def get_model():
    """Download trained model with error handling."""
    try:
        if not os.path.exists(MODEL_PATH):
            if training_status["running"]:
                logger.info("Model requested but training is still in progress.")
                raise HTTPException(status_code=425, detail="Training still in progress. Model not ready yet.")
            elif training_status["error"]:
                logger.error(f"Model requested but training failed: {training_status['error']}")
                raise HTTPException(status_code=500, detail=f"Training failed: {training_status['error']}")
            else:
                logger.info("Model requested but not found. Training has not started.")
                raise HTTPException(status_code=404, detail="Model not found. Please start training first.")

        # Get file size for logging
        file_size = os.path.getsize(MODEL_PATH)
        logger.info(f"Serving trained model file: {MODEL_PATH} ({file_size} bytes)")

        return FileResponse(
            MODEL_PATH,
            media_type="application/octet-stream",
            filename="trained_model.keras",
            headers={"Content-Length": str(file_size)}
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to serve trained model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to serve model: {str(e)}")

@app.get("/health/")
async def health_check():
    """Health check endpoint."""
    logger.info("Health check requested.")
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "data_files_exist": {
            "data.npy": os.path.exists(os.path.join(DATA_DIR, "data.npy")),
            "labels.npy": os.path.exists(os.path.join(DATA_DIR, "labels.npy"))
        },
        "model_exists": os.path.exists(MODEL_PATH)
    }

if __name__ == "__main__":
    logger.info(f"Starting ML Training Server at {server}:{port}")
    uvicorn.run(
        "server:app",
        host=server,
        port=port,
        reload=False,
        log_level="warning"
    )
