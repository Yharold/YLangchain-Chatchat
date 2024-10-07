from config.server_config import WEBUI_SERVER


def get_webui_address():
    host = WEBUI_SERVER["host"]
    port = WEBUI_SERVER["port"]
    return host, port
