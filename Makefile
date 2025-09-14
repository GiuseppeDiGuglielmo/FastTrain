.PHONY: help server client clean stop

HOSTNAME = $(shell hostname)
CLIENT ?= 131.225.220.156
SERVER ?= 131.225.220.30

help:
	@echo "FastTrain - Available commands:"
	@echo ""
	@echo "  make server      - Start the training server"
	@echo "  make client      - Run the client"
	@echo "  make clean       - Clean generated files"
	@echo "  make stop        - Stop running processes"
.PHONY: help

tunnel-to-server:
	@echo "INFO: Setting up SSH tunnel to $(SERVER):8000"
	@ssh -N -g -L 8000:$(SERVER):8000 gdg@$(SERVER)
.PHONY: tunnel-to-server

server:
	@echo "INFO: Starting FastTrain server on $(SERVER):8000"
	@stdbuf -oL -eL python server.py 2> >(grep -vE "\+ptx[0-9]+")
.PHONY: server

client:
	@echo "INFO: Starting FastTrain client on $(CLIENT)"
	@python client.py
.PHONY: client

clean:
	@echo "INFO: Cleaning generated files..."
	rm -f *.pyc
	rm -rf __pycache__/
	rm -f data.npy labels.npy
	rm -f model_downloaded.keras trained_model.keras
	rm -f server.log
	rm -rf uploaded_data/
	rm -rf model_1/
	@echo "Cleanup complete!"
.PHONY: clean

stop:
	@echo "INFO: Stopping FastTrain processes..."
	pkill -f "python server.py" || echo "No server processes found"
	pkill -f "uvicorn server:app" || echo "No uvicorn processes found"
.PHONY: stop