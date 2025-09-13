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

tunnel-to-server:
	@ssh -N -g -L 8000:correlator3.fnal.gov:8000 gdg@correlator3.fnal.gov

server:
	@echo "Starting FastTrain server on https://${HOSTNAME}:8000 ..."
	LD_PRELOAD="$(CONDA_PREFIX)/lib/libgomp.so.1" python server.py --client ${CLIENT} --server ${SERVER}


client:
	@echo "Running FastTrain client on https://${HOSTNAME}:8000 ..."
	python client.py --client ${CLIENT} --server ${SERVER}

clean:
	@echo "Cleaning generated files..."
	rm -f *.pyc
	rm -rf __pycache__/
	rm -f data.npy labels.npy
	rm -f model_downloaded.keras trained_model.keras
	rm -f server.log
	rm -rf uploaded_data/
	rm -rf model_1/
	@echo "Cleanup complete!"

stop:
	@echo "Stopping FastTrain processes..."
	pkill -f "python server.py" || echo "No server processes found"
	pkill -f "uvicorn server:app" || echo "No uvicorn processes found"
