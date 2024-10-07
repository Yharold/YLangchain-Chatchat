from config.server_config import CONTROLLER_SERVER


def get_controller_address():
    host = CONTROLLER_SERVER["host"]
    port = CONTROLLER_SERVER["port"]
    return host, port
