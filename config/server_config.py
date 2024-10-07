import sys


DEFAULT_BIND_HOST = "0.0.0.0" if sys.platform != "win32" else "127.0.0.1"
CONTROLLER_SERVER = {"host": DEFAULT_BIND_HOST, "port": 9527}
MODEL_WORKER_SERVER = {"host": DEFAULT_BIND_HOST, "port": 9528}
FASTAPI_SERVER = {"host": DEFAULT_BIND_HOST, "port": 9529}
WEBUI_SERVER = {"host": DEFAULT_BIND_HOST, "port": 9530}
