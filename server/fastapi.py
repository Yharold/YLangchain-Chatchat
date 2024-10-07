from fastapi import FastAPI
from config.server_config import FASTAPI_SERVER

app = FastAPI()


def get_fastapi_address():
    host = FASTAPI_SERVER["host"]
    port = FASTAPI_SERVER["port"]
    return host, port

