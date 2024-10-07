from config.server_config import MODEL_WORKER_SERVER


def get_worker_address():
    host = MODEL_WORKER_SERVER["host"]
    port = MODEL_WORKER_SERVER["port"]
    return host, port
