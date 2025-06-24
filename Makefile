.PHONY: help server client clean stop

help:
	@echo "FastTrain - Available commands:"
	@echo ""
	@echo "  make server      - Start the training server"
	@echo "  make client      - Run the client"
	@echo "  make clean       - Clean generated files"
	@echo "  make stop        - Stop running processes"

server:
	@echo "Starting FastTrain server on http://127.0.0.1:8000"
	python server.py

client:
	@echo "Running FastTrain client..."
	python client.py

clean:
	@echo "Cleaning generated files..."
	rm -f *.pyc
	rm -rf __pycache__/
	rm -f data.npy labels.npy
	rm -f model_downloaded.h5 trained_model.h5
	rm -f server.log
	rm -rf uploaded_data/
	rm -rf model_1/
	@echo "Cleanup complete!"

stop:
	@echo "Stopping FastTrain processes..."
	pkill -f "python server.py" || echo "No server processes found"
	pkill -f "uvicorn server:app" || echo "No uvicorn processes found"
