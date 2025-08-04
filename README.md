# FastTrain 🚂

When training a model on data collected from a device (`client`) that is too slow for efficient processing, you can offload the training to a more powerful `server` to accelerate the process. Here is **FastTrain** 🚂. While there are likely other libraries that support this decoupling more efficiently, this is a first proof of concept, where the `client` and `server` are connected over an Ethernet LAN (or even the Internet).

The `client` and `server` could be any platforms/devices.

<p align="center">
  <img src="doc/client_server.png" width="1024px">
  <img src="doc/demo.gif" width="1024px">
</p>

## Quick Start

```bash
# Install dependencies
pip install tensorflow fastapi uvicorn scikit-learn numpy requests

# Start server
python server.py
# or
make server

# Run client (in another terminal)
python client.py
# or
make client

# Clean directory
make clean
```

