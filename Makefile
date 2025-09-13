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
	#LD_PRELOAD="$(CONDA_PREFIX)/lib/libgomp.so.1" python server.py --client ${CLIENT} --server ${SERVER}
	@echo "Starting FastTrain server on http://127.0.0.1:8000"
	@stdbuf -oL -eL python server.py 2> >(grep -vE "\+ptx[0-9]+")
	#LD_PRELOAD="$(CONDA_PREFIX)/lib/libgomp.so.1" python server.py

client:
	@echo "Running FastTrain client..."
	@python client.py

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
